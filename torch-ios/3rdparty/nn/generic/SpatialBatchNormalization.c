#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialBatchNormalization.c"
#else

static int nn_(SpatialBatchNormalization_updateOutput)(lua_State *L)
{
    THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
    THTensor *running_mean = luaT_getfieldcheckudata(L, 1, "running_mean", torch_Tensor);
    THTensor *running_var = luaT_getfieldcheckudata(L, 1, "running_var", torch_Tensor);
    double eps = 0.00001;

    
    
    
    
    
    
    THTensor_(resizeAs)(output, input);
  long nInput = THTensor_(size)(input, 1);
  long f,n = THTensor_(nElement)(input) / nInput;

  for (f = 0; f < nInput; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *out = THTensor_(newSelect)(output, 1, f);

    real mean, invstd;

    mean = THTensor_(get1d)(running_mean, f);
    invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
    

    // compute output
    real w = 1;
    real b = 0;

    TH_TENSOR_APPLY2(real, in, real, out,
      *out_data = (real) (((*in_data - mean) * invstd) * w + b););

    THTensor_(free)(out);
    THTensor_(free)(in);
  }
    
  return 1;
}



static const struct luaL_Reg nn_(SpatialBatchNormalization__) [] = {
    {"SpatialBatchNormalization_updateOutput", nn_(SpatialBatchNormalization_updateOutput)},
    {NULL, NULL}
};


static void nn_(SpatialBatchNormalization_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, nn_(SpatialBatchNormalization__), "nn");
    lua_pop(L,1);
}

#endif
