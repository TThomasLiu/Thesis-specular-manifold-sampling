#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <enoki/morton.h>

#include <mitsuba/core/progress.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <mutex>

#include <mitsuba/render/flow_ms.h>
#include <mitsuba/render/torch_bridge/torch_bridge.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * This is mostly a standard path tracer, but augmented with the specular
 * manifold sampling (SMS) strategy for (multi-bounce) caustics from (near)
 * delta reflections or refractions.
 *
 * This integrator is used for Figure 18 in the paper.
 *
 * The integration into the path tracer is more or less specialized for that
 * scene currently. Extensions such as (approximative) MIS to robustly transition
 * to classical sampling strategies in cases of high roughness or large light
 * sources are an interesting direction for future work.
 *
 * Several options are available to fine-tune which variation of SMS should be
 * performed. See the SMSConfig struct in 'manifold.h'. The number of bounces
 * for the caustics should also be set there explicitly.
 *
 * We can also compare against Manifold Next Event Estimation (MNEE) with it
 * by setting sms_config.mnee_init=true, sms_config.biased=true,
 * sms_config.max_trials=1, sms_config.halfvector_constraints=true, and
 * biased_mnee=true/false depending on which flavour of MNEE should be used.
 *
 * This multi-bounce implementation doesn't support specular manifold sampling
 * with glossy/rough materials, but it could be extended to that case as well.
 *
 */
template <typename Float, typename Spectrum>
class MultiScatterFlowPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_block_size, m_samples_per_pass, m_timeout, m_hide_emitters, m_glint_diff_scale_factor_clamp, should_stop, m_render_timer)
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium, ImageBlock, Film)
    using SpecularManifold = SpecularManifold<Float, Spectrum>;
    using FlowSpecularManifoldMultiScatter = FlowSpecularManifoldMultiScatter<Float, Spectrum>;
    using MonteCarloIntegrator = MonteCarloIntegrator<Float, Spectrum>;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;

protected:
    /* The integration of SMS is pretty straight forward in the multi-bounce
       case as well---but Manifold Next Event Estimation unfortunately gets
       a bit more tricky to implement.
       For each random light connection we want to check if that light path could
       have also been generated with MNEE, so we need to keep some additional
       state around. This helper struct here takes care of this. */
    struct WaveVariables{
        // write position
        Vector2f position_sample;
        Spectrum ray_weight;

        // sample input
        ref<Sampler> sampler;
        RayDifferential3f ray_;
        RayDifferential3f ray;
        Mask active = true;

        // wave front variables
        bool specular_camera_path = true; 
        float sample_weight = 1.f;
        int model_index = 0;
        int compact_index = 0;
        Float eta = 1.f;
        
        bool enable= true;
        Vector3f flow_direction;
        float flow_weight = 1.0f;

        Spectrum throughput;
        SurfaceInteraction3f si;
        EmitterInteraction ei;
        
        // output
        Spectrum result;
        Mask valid_ray;

        void reset() {
            ray_weight = 1.f;
            throughput = 1.f;
            result = 0.f;
            eta = 1.f;
            specular_camera_path = true;
            flow_direction = Vector3f(0.f, 0.f, 0.f);
            flow_weight = 1.0f;
            enable = true;
            sample_weight = 1.f;
            model_index = 0;
            compact_index = 0;
            valid_ray = true;
            active = true;
        }
    };

public:
    MultiScatterFlowPathIntegrator(const Properties &props) : Base(props) {
        m_sms_config = SMSConfig();

        m_sms_config.bounces                = props.int_("bounces", 2);
        m_sms_config.remove_pt_direct_hit   = props.bool_("remove_pt_direct_hit", false);

        m_visnet_sms_config.visnet_enable = props.bool_("visnet_enable", false);
        m_visnet_sms_config.visnet_threshold = props.float_("visnet_threshold", 0.4f);

        m_rmax = props.float_("rmax", 3.0f);
        m_noise_std = props.float_("noise_std", 0.01f);
        
        std::string model_device = props.string("model_device", "gpu");
        if (model_device == "gpu") {
            m_device_gpu = true;
        } else if (model_device == "cpu") {
            m_device_gpu = false;
        } else {
            Throw("Invalid model_device '%s'! Should be either 'gpu' or 'cpu'.", model_device);
        }
    }

    bool render(Scene *scene, Sensor *sensor) override {
        auto shapes = scene->caustic_casters_multi_scatter();
        if (m_visnet_sms_config.visnet_enable){
            for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
                const ShapePtr specular_shape = shapes[shape_idx];
                if (!specular_shape->visnet_model_path().empty()) {
                    torch_load_visnet_model(specular_shape->visnet_model_path().c_str(), m_device_gpu);
                    m_to_model = specular_shape->to_object();

                    if(!specular_shape->flow_model_path().empty()){
                        torch_load_flow_model(specular_shape->flow_model_path().c_str(), m_device_gpu);

                        // load m_to_world matrix to flat vector
                        const auto& m4 = specular_shape->to_world().matrix;
                        m_flat_to_world.resize(9);
                        for (int i = 0; i < 3; i++) {
                            for (int j = 0; j < 3; j++) {
                                m_flat_to_world[i * 3 + j] = m4(i, j);
                            }
                        }
                    }
                }
            }
        }

        bool result = sequential_block_render(scene, sensor);
        return result;
    }

    static EmitterInteraction sample_emitter(const SurfaceInteraction3f &si, const std::vector<ref<Emitter>> emitters, ref<Sampler> sampler){
        EmitterInteraction ei;
        Spectrum spec = 0.f;
        Float emitter_sample = sampler->next_1d();
        Float emitter_pdf = 1.f / emitters.size();
        UInt32 index = min(UInt32(emitter_sample * (ScalarFloat) emitters.size()), (uint32_t) emitters.size()-1);
        const EmitterPtr emitter = gather<EmitterPtr>(emitters.data(), index);
        ei.emitter = emitter;


        if (ei.is_area()) {
            const ShapePtr shape = emitter->shape();
            PositionSample3f ps = shape->sample_position(si.time, sampler->next_2d());
            if (ps.pdf > 0) {
                SurfaceInteraction3f si_emitter;
                si_emitter.p = ps.p;
                si_emitter.wi = Vector3f(0.f, 0.f, 1.f);
                si_emitter.wavelengths = si.wavelengths;
                si_emitter.time = si.time;

                spec = emitter->eval(si_emitter) / ps.pdf;

                ei.p = ps.p;
                ei.n = ps.n;
                ei.d = normalize(ps.p - si.p);
                ei.pdf = ps.pdf;
            }
        }else if(ei.is_point()){
            auto [ds, spec_] = emitter->sample_direction(si, sampler->next_2d());
            ei.p = ds.p;
            ei.d = ds.d;
            ei.n = ei.d;
            ei.pdf = ds.pdf;
            // Remove solid angle conversion factor. This will be accounted for later in the geometric term computation.
            spec = spec_ * ds.dist*ds.dist;
        }else{
            throw std::runtime_error("Unsupported emitter type for sample_emitter.");
        }

        ei.pdf *= emitter_pdf;
        ei.weight = spec * rcp(emitter_pdf);
        return ei;
    };

    Float gaussian_weight_2d(Float r, Float std_dev, Float r_max) const {
        using enoki::exp;
        using enoki::select;

        Float r_max_scaled = r_max * std_dev;

        Float eps_c = exp(Float(-0.5) * (r_max_scaled / std_dev) * (r_max_scaled / std_dev));  // tail mass beyond r_max
        Float norm_const = Float(2) * math::Pi<Float> * std_dev * std_dev * (Float(1) - eps_c);

        Float diff = (r) / std_dev;
        Float weight = exp(Float(-0.5) * diff * diff) / norm_const;

        return select(r <= r_max_scaled, weight, Float(0));  // Enoki 向量化下的條件選擇,取代 np.where
    }

    Spectrum trace_photon(const Scene* scene, WaveVariables& variable_set, int depth) const {
        Spectrum throughput(1.0f);
        Ray3f ray(variable_set.ei.p, variable_set.flow_direction, variable_set.si.time, variable_set.si.wavelengths);

        // if(variable_set.ei.is_area()){
        //     throughput *= dot(variable_set.ei.n, variable_set.flow_direction);
        // }

        SurfaceInteraction3f si;
        bool success= false;
        for(int i = 0; i < depth; ++i){
            si = scene->ray_intersect(ray);
            const ShapePtr shape = si.shape;

            
            if(!si.is_valid()){
                return 0.f;
            }

            // // geometric term for emitter
            // if(i == 0){
            //     auto emitter_dist = norm(variable_set.ei.p - si.p);
            //     throughput *= rcp(emitter_dist * emitter_dist);
            // }


            // check if the shape is a caustic receiver
            if(shape->is_caustic_receiver()){
                // get si distance to the receiver
                Float dist = norm(si.p - variable_set.si.p);
                Float weight = gaussian_weight_2d(dist, 0.01f, 3.f);
                variable_set.flow_weight *= weight;

                if (i != 2) {
                    return 0.f;
                }
                success = true;
                break;
            }
            
            if(!shape->is_caustic_caster_multi_scatter() &&
            !shape->is_caustic_bouncer()){
                return 0.f;
            }


            si.compute_partials(ray);

            BSDFContext ctx;
            ctx.sampler = variable_set.sampler;
            BSDFPtr bsdf = si.bsdf(ray);

            // Sample a new direction
            auto [bs, bsdf_weight] = bsdf->sample(ctx, si, variable_set.sampler->next_1d(), variable_set.sampler->next_2d());
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bs.wo, si.wi);
            throughput = throughput * bsdf_weight;
            if (all(eq(throughput, 0.f))){
                return 0.f;
            }

            // Update the ray for the next bounce
            ray = si.spawn_ray(si.to_world(bs.wo));
        }

        if(!success){
            return 0.0f;
        }

        BSDFPtr bsdf = variable_set.si.bsdf(variable_set.ray);
        BSDFContext ctx;
        ctx.sampler = variable_set.sampler;

        auto bsdf_weight = bsdf->eval(ctx, variable_set.si, ray.d);
        bsdf_weight = variable_set.si.to_world_mueller(bsdf_weight, ray.d, variable_set.si.wi);
        throughput = throughput * bsdf_weight;
        return throughput;
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        
        Throw("original sample method not implemented.");
        return { 0.f, 0.f };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("FlowMultiScatterSMSPathIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()
protected:
    SMSConfig m_sms_config;
    VisnetSMSConfig m_visnet_sms_config;
    bool m_device_gpu;
    bool m_biased_mnee;      // Make MNEE biased by filtering out caustic paths that can't be sampled with it
    ScalarTransform4f m_to_model;
    float m_rmax;
    float m_noise_std;
    
    std::vector<float> m_flat_to_world;

    // sample
    void bounce_step(WaveVariables& variable_set, int depth,  const Medium *medium, const Scene *scene) const {
        if(!variable_set.active){
            return;
        }
        
        if(depth == 0){
            variable_set.ray = variable_set.ray_;
            variable_set.eta = 1.f;
            variable_set.throughput = 1.f;
            variable_set.result = 0.f;
            variable_set.specular_camera_path = true;   // To capture emitters visible direcly through purely specular reflection/refractions

            // ---------------------- First intersection ----------------------

            variable_set.si = scene->ray_intersect(variable_set.ray);
            variable_set.valid_ray = variable_set.si.is_valid();
            EmitterPtr emitter = variable_set.si.emitter(scene);

            if (emitter) {
                variable_set.result += emitter->eval(variable_set.si);
            }
            return;
        }

        // ------------------ Possibly terminate path -----------------

        if (!variable_set.si.is_valid()){
            variable_set.active = false;
            return;
        }
        
        variable_set.si.compute_partials(variable_set.ray);

        if (depth > m_rr_depth) {
            Float q = min(hmax(depolarize(variable_set.throughput)) * sqr(variable_set.eta), .95f);
            if (variable_set.sampler->next_1d() > q){
                variable_set.active = false;
                return;
            }
            variable_set.throughput *= rcp(q);
        }

        // --------------- Specular Manifold Sampling -----------------

        
        bool on_caustic_caster = variable_set.si.shape->is_caustic_caster_multi_scatter() ||
                                    variable_set.si.shape->is_caustic_bouncer();

        if (variable_set.si.shape->is_caustic_receiver() && !on_caustic_caster &&
            (m_max_depth < 0 || depth + m_sms_config.bounces < m_max_depth)&& variable_set.enable) {

            // TODO: caustic rendering logics

            Spectrum photon_result = trace_photon(scene, variable_set, m_sms_config.bounces + 1);
            
            variable_set.result += variable_set.throughput * photon_result * variable_set.flow_weight * variable_set.ei.weight;
        }

        // --------------------- Emitter sampling ---------------------

        BSDFContext ctx;
        ctx.sampler = variable_set.sampler;
        BSDFPtr bsdf = variable_set.si.bsdf(variable_set.ray);

        /* As usual, emitter sampling only makes sense on Smooth BSDFs
            that can be evaluated.
            Additionally, filter out:
            - paths that we could previously sample with SMS
            - paths that are even harder to sample, e.g. paths bouncing
                off several caustic casters before hitting the light.
            As a result, we only do emitter sampling on non-caustic
            casters---with the exception of the first bounce where we might
            see a direct (glossy) reflection of a light source this way.

            Note: of course, SMS might not always be the optimal sampling
            strategy. For example, when rough surfaces are involved it
            would be still better to do emitter sampling.
            A way of incoorporating MIS with all of this would be super
            useful. */
            
        if (has_flag(bsdf->flags(), BSDFFlags::Smooth) &&
            !on_caustic_caster) {
            /* In case we didn't scatter off a caustic receiver before
                or aren't interacting with a caustic caster now, do
                emitter sampling as usual. */
            auto [ds, emitter_weight] = scene->sample_emitter_direction(variable_set.si, variable_set.sampler->next_2d(), true);
            if (ds.pdf != 0.f) {
                // Query the BSDF for that emitter-sampled direction
                Vector3f wo = variable_set.si.to_local(ds.d);
                Spectrum bsdf_val = bsdf->eval(ctx, variable_set.si, wo);
                bsdf_val = variable_set.si.to_world_mueller(bsdf_val, -wo, variable_set.si.wi);

                // Determine density of sampling that same direction using BSDF sampling
                Float bsdf_pdf = bsdf->pdf(ctx, variable_set.si, wo);
                Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                variable_set.result += mis * variable_set.throughput * bsdf_val * emitter_weight;
            }
        }

        // ----------------------- BSDF sampling ----------------------

        // Sample BSDF * cos(theta)
        auto [bs, bsdf_weight] = bsdf->sample(ctx, variable_set.si, variable_set.sampler->next_1d(),
                                                variable_set.sampler->next_2d());
        bsdf_weight = variable_set.si.to_world_mueller(bsdf_weight, -bs.wo, variable_set.si.wi);

        variable_set.throughput = variable_set.throughput * bsdf_weight;
        variable_set.eta *= bs.eta;
        if (!has_flag(bs.sampled_type, BSDFFlags::Delta)) {
            variable_set.specular_camera_path = false;
        }

        if (all(eq(variable_set.throughput, 0.f))){
            variable_set.active = false;
            return;
        }
        
        
        // Intersect the BSDF ray against the scene geometry
        variable_set.ray = variable_set.si.spawn_ray(variable_set.si.to_world(bs.wo));
        SurfaceInteraction3f si_bsdf = scene->ray_intersect(variable_set.ray);
        EmitterPtr emitter = si_bsdf.emitter(scene);

        // Hit emitter after BSDF sampling
        if (emitter && !m_sms_config.remove_pt_direct_hit) {
            /* With the same reasoning as in the emitter sampling case,
                filter out some of the light paths here.
                Again, this is unfortunately not robust in all cases,
                for large light sources, BSDF sampling would be more
                appropriate than relying purely on SMS. */
            if (!on_caustic_caster || variable_set.specular_camera_path) {
                /* Only do BSDF sampling in usual way if we don't interact
                    with a caustic caster now. */

                // Evaluate the emitter for that direction
                Spectrum emitter_val = emitter->eval(si_bsdf);

                /* Determine probability of having sampled that same
                    direction using emitter sampling. */
                DirectionSample3f ds(si_bsdf, variable_set.si);
                ds.object = emitter;
                Float emitter_pdf = select(!has_flag(bs.sampled_type, BSDFFlags::Delta),
                                            scene->pdf_emitter_direction(variable_set.si, ds),
                                            0.f);
                Float mis = mis_weight(bs.pdf, emitter_pdf);
                variable_set.result += mis * variable_set.throughput * emitter_val;
            }
        }

        variable_set.si = std::move(si_bsdf);


    }

    // sampling loop
    void sampling_loop(
        int block_id,
        const Scene *scene,
        Sensor *sensor,
        Sampler *sampler,
        ImageBlock *block,
        Float *aovs,
        size_t sample_count_ = size_t(-1)) const
    {
        ThreadEnvironment env;
        uint32_t pixel_count  = (uint32_t)(m_block_size * m_block_size);
        uint32_t sample_count = (uint32_t)(sample_count_ == (size_t) -1
            ? sampler->sample_count()
            : sample_count_);
        ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());
        diff_scale_factor = max(diff_scale_factor, m_glint_diff_scale_factor_clamp);


        // sample
        static std::vector<int> compact_map (pixel_count);
        static std::vector<WaveVariables> wave_variables(pixel_count);
        static std::vector<float> model_inputs (pixel_count * 6);
        static std::vector<float> temp_model_inputs (pixel_count * 6);
        static std::vector<int> model_outputs (pixel_count);
        
        static std::vector<float> flow_direction (pixel_count * 3);
        static std::vector<float> flow_direction_weight (pixel_count);
        
        std::atomic<int>  SMS_enable_count;
        std::atomic<int>  active_count;

        active_count = pixel_count;
        
        // assign sampler
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, pixel_count, 1),
            [&](const tbb::blocked_range<size_t> &range) {
                    for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                        WaveVariables& variable_set = wave_variables[i];
                        variable_set.sampler = sampler->clone();
                        variable_set.sampler->seed(block_id * pixel_count * sample_count + i);
                    }
            }
        );

        for(int sample_idx = 0 ; sample_idx < sample_count; sample_idx++){
            // ray init
            {
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, pixel_count, 1),
                    [&](const tbb::blocked_range<size_t> &range) {
                        ScopedSetThreadEnvironment set_env(env);
                        
                        for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                            WaveVariables& variable_set = wave_variables[i];
                            // variable initialization
                            variable_set.reset();
                            compact_map[i] = i;

                            variable_set.position_sample = enoki::morton_decode<ScalarPoint2u>(i);
                            
                            if (any(variable_set.position_sample >= block->size())){
                                variable_set.active = false;
                            }
                            variable_set.position_sample += block->offset();

                            // MonteCarloIntegrator::render_sample
                            {
                                variable_set.position_sample += variable_set.sampler->next_2d(variable_set.active);

                                Point2f aperture_sample(.5f);
                                if (sensor->needs_aperture_sample())
                                    aperture_sample = variable_set.sampler->next_2d(variable_set.active);

                                Float time = sensor->shutter_open();
                                if (sensor->shutter_open_time() > 0.f)
                                    time += variable_set.sampler->next_1d(variable_set.active) * sensor->shutter_open_time();

                                Float wavelength_sample = variable_set.sampler->next_1d(variable_set.active);

                                Vector2f adjusted_position =
                                    (variable_set.position_sample - sensor->film()->crop_offset()) /
                                    sensor->film()->crop_size();

                                auto [ray, ray_weight] = sensor->sample_ray_differential(
                                    time, wavelength_sample, adjusted_position, aperture_sample);

                                ray.scale_differential(diff_scale_factor);

                                variable_set.ray_ = ray;
                                variable_set.ray_weight = ray_weight;
                            }
                        }
                    }
                );
            }

            // depth loop
            for(int depth = 0; depth < m_max_depth; depth++){
                bool visnet_enable = false;
                
                if(depth != 0){
                    // compact
                    {
                        int old_active_count = active_count;
                        ScopedPhase scope_phase(ProfilerPhase::Compact);
                        active_count = 0;
                        for(int i = 0; i < old_active_count; ++i){
                            WaveVariables& variable_set = wave_variables[compact_map[i]];
                            if(variable_set.active){
                                compact_map[active_count] = compact_map[i];
                                ++active_count;
                            }
                        }
                    }
                    
                    // ei sampling
                    {               
                        SMS_enable_count = 0;
                        // bool sms_depth_check = (m_max_depth < 0 || depth + m_sms_config.bounces < m_max_depth);
                        tbb::parallel_for(
                            tbb::blocked_range<size_t>(0, active_count, 1),
                            [&](const tbb::blocked_range<size_t> &range) {
                                ScopedSetThreadEnvironment set_env(env);
                                ScopedPhase scope_phase(ProfilerPhase::TorchVariableConvert);
                                for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                                    WaveVariables& variable_set = wave_variables[compact_map[i]];
                                    variable_set.ei = SpecularManifold::sample_emitter_interaction(variable_set.si, scene->caustic_emitters_multi_scatter(), variable_set.sampler);
                                    
                                    // test sample ei
                                    sample_emitter(variable_set.si, scene->caustic_emitters_multi_scatter(), variable_set.sampler);
    
                                    if (!m_visnet_sms_config.visnet_enable) continue;
    
                                    if (variable_set.active && 
                                        variable_set.si.is_valid() && 
                                        variable_set.si.shape->is_caustic_receiver() &&
                                        (m_max_depth < 0 || depth + m_sms_config.bounces < m_max_depth))
                                    {
                                        
                                        variable_set.model_index = SMS_enable_count.fetch_add(1, std::memory_order_relaxed);
                                    }else{
                                        variable_set.model_index = -1;
                                        continue;
                                    }
    
                                    int index = variable_set.model_index * 6;
                                    Point3f p_si = variable_set.si.p;
                                    Point3f p_ei = variable_set.ei.p;
                                    p_si = m_to_model.transform_affine(p_si);
                                    p_ei = m_to_model.transform_affine(p_ei);

                                    memcpy(&model_inputs[index], &p_ei, sizeof(float) * 3);
                                    memcpy(&model_inputs[index + 3], &p_si, sizeof(float) * 3);
                                }
                            }
                        );
                    }

                    // model run
                    {
                        if (SMS_enable_count > 0) {
                            ScopedPhase scope_phase(ProfilerPhase::TorchModelRun);
                            torch_visnet_forward(model_inputs.data(), model_outputs.data(), SMS_enable_count, m_visnet_sms_config.visnet_threshold);

                            // create data for flow model
                            int flow_enable_count = 0;
                            int visnet_count = 0;
                            for(int i = 0 ; i < active_count; i++){
                                WaveVariables& variable_set = wave_variables[compact_map[i]];
                                if(variable_set.model_index == -1) continue;

                                if(model_outputs[variable_set.model_index]){
                                    memcpy(&temp_model_inputs[flow_enable_count * 6], &model_inputs[variable_set.model_index * 6], sizeof(float) * 6);
                                    variable_set.model_index = flow_enable_count;
                                    variable_set.enable = true;
                                    flow_enable_count++;
                                }else{
                                    variable_set.enable = false;
                                    variable_set.model_index = -1;
                                }

                                visnet_count++;
                            }
                            model_inputs.swap(temp_model_inputs);



                            std::cout<<"flow_enable_count: "<<flow_enable_count<<std::endl;
                            if(flow_enable_count > 0){
                                torch_flow_forward(model_inputs.data(), m_flat_to_world.data(), flow_direction.data(), flow_direction_weight.data(), flow_enable_count, 10);
                                std::cout<<"called flow model"<<std::endl;
                            }

                        }
                    }
                }

                // bounce
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, active_count, 1),
                    [&](const tbb::blocked_range<size_t> &range) {
                        ScopedSetThreadEnvironment set_env(env);
                        scoped_flush_denormals flush_denormals(true);

                        for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                            WaveVariables& variable_set = wave_variables[compact_map[i]];
                            
                            if(variable_set.model_index != -1){
                                int model_index = variable_set.model_index;
                                variable_set.flow_direction = Vector3f(flow_direction[model_index * 3], flow_direction[model_index * 3 + 1], flow_direction[model_index * 3 + 2]);
                                variable_set.flow_weight = flow_direction_weight[model_index];
                            }

                            bounce_step(variable_set, depth, sensor->medium(), scene);
                        }
                    }
                );
            }
            
            // put to block
            {
                for (int i = 0; i < pixel_count && !should_stop(); ++i) {
                    WaveVariables& variable_set = wave_variables[i];
                    variable_set.result = variable_set.ray_weight * variable_set.result;

                    UnpolarizedSpectrum spec_u = depolarize(variable_set.result);

                    Color3f xyz;
                    if constexpr (is_monochromatic_v<Spectrum>) {
                        xyz = spec_u.x();
                    } else if constexpr (is_rgb_v<Spectrum>) {
                        xyz = srgb_to_xyz(spec_u, true);
                    } else {
                        static_assert(is_spectral_v<Spectrum>);
                        xyz = spectrum_to_xyz(spec_u, variable_set.ray_.wavelengths, true);
                    }

                    aovs[0] = xyz.x();
                    aovs[1] = xyz.y();
                    aovs[2] = xyz.z();
                    aovs[3] = select(variable_set.valid_ray, Float(1.f), Float(0.f));
                    aovs[4] = 1.f;

                    block->put(variable_set.position_sample, aovs, true);
                }
            }
        }
    }

    void parallel_pixel_render_block(
        int block_id,
        const Scene *scene,
        Sensor *sensor,
        Sampler *sampler,
        ImageBlock *block,
        Float *aovs,
        size_t sample_count_ = size_t(-1)
    ) const {
        block->clear();

        if constexpr (!is_array_v<Float>) {
            sampling_loop(block_id, scene, sensor, sampler, block, aovs, sample_count_);

        } else if constexpr (is_array_v<Float> && !is_cuda_array_v<Float>) {
            ENOKI_MARK_USED(scene);
            ENOKI_MARK_USED(sensor);
            ENOKI_MARK_USED(aovs);
            Throw("Not implemented for arrays.");
            
        } else {
            ENOKI_MARK_USED(scene);
            ENOKI_MARK_USED(sensor);
            ENOKI_MARK_USED(aovs);
            Throw("Not implemented for CUDA arrays.");
        }
    }


    bool sequential_block_render(Scene *scene, Sensor *sensor) {
        ScopedPhase sp(ProfilerPhase::Render);
        MonteCarloIntegrator::m_stop = false;

        ref<Film> film = sensor->film();
        ScalarVector2i film_size = film->crop_size();

        size_t total_spp        = sensor->sampler()->sample_count();
        size_t samples_per_pass = (m_samples_per_pass == (size_t) -1)
                                ? total_spp : std::min((size_t) m_samples_per_pass, total_spp);
        if ((total_spp % samples_per_pass) != 0)
            Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
                total_spp, samples_per_pass);

        size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

        std::vector<std::string> channels = MonteCarloIntegrator::aov_names();
        bool has_aovs = !channels.empty();

        // Insert default channels and set up the film
        for (size_t i = 0; i < 5; ++i)
            channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
        film->prepare(channels);

        if constexpr (!is_cuda_array_v<Float>) {
            /// Render on the CPU using a spiral pattern
            size_t n_threads = __global_thread_count;
            Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
                film_size.x(), film_size.y(),
                total_spp, total_spp == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
                n_threads, n_threads == 1 ? "" : "s");

            if (m_timeout > 0.f)
                Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

            // Find a good block size to use for splitting up the total workload.
            if (m_block_size == 0) {
                uint32_t block_size = MTS_BLOCK_SIZE;
                while (true) {
                    if (block_size == 1 || hprod((film_size + block_size - 1) / block_size) >= n_threads)
                        break;
                    block_size /= 2;
                }
                m_block_size = block_size;
            }

            Spiral spiral(film, m_block_size, n_passes);

            // ThreadEnvironment env;
            ref<ProgressReporter> progress = new ProgressReporter("Rendering");
            // std::mutex mutex;

            // Total number of blocks to be handled, including multiple passes.
            size_t total_blocks = spiral.block_count() * n_passes,
                blocks_done = 0;

            m_render_timer.reset();
            
            // Process each block in single thread
            for(size_t idx = 0; idx < total_blocks && !should_stop(); ++idx) {
                // std::cout<<"Rendering block "<<idx<<"/"<<total_blocks<<std::endl;
                ref<ImageBlock> block = new ImageBlock(m_block_size, channels.size(),
                                 film->reconstruction_filter(),
                                 !has_aovs);
                // scoped_flush_denormals flush_denormals(true);
                std::unique_ptr<Float[]> aovs(new Float[channels.size()]);

                auto [offset, size, block_id] = spiral.next_block();
                Assert(hprod(size) != 0);
                block->set_size(size);
                block->set_offset(offset);

                // Ensure that the sample generation is fully deterministic

                parallel_pixel_render_block(block_id, scene, sensor, sensor->sampler(), block,
                aovs.get(), samples_per_pass);
                
                film->put(block);
                
                // report progress
                blocks_done++;
                progress->update(blocks_done / (ScalarFloat) total_blocks);
            }

            if (!MonteCarloIntegrator::m_stop) {
                Log(Info, "Rendering finished. %s ms", m_render_timer.value());

                if (m_timeout > 0) {
                    Float spp_f = blocks_done;
                    spp_f /= spiral.block_count();
                    int spp = int(floor(spp_f));

                    Log(Info, "Rendering finished. Computed %d spp and took %s.",
                        spp, util::time_string(m_render_timer.value(), true));
                        
                    Log(Info, "Rendering finished. %f ms / spp.",
                        m_render_timer.value() / spp);
                } else {
                    Log(Info, "Rendering finished. (took %s)",
                        util::time_string(m_render_timer.value(), true));
                }
            }

        } else {
            Throw("Not implemented for CUDA arrays.");
        }

        return !MonteCarloIntegrator::m_stop;
    }
};

MTS_IMPLEMENT_CLASS_VARIANT(MultiScatterFlowPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MultiScatterFlowPathIntegrator, "Multi-Bounce Flow Path Tracer integrator");
NAMESPACE_END(mitsuba)
