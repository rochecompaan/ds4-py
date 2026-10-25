from ds4py._native.bindings import Binding

BINDINGS = (
    Binding("ds4_engine_layer_count", "int ds4_engine_layer_count(ds4_engine *e)"),
    Binding(
        "ds4_engine_sampling_defaults",
        "void ds4_engine_sampling_defaults(ds4_engine *e, float *temperature, int *top_k, float *top_p, float *min_p)",
        required=False,
    ),
)
