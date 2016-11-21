#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialAveragePooling.c"
#else

static int nn_(SpatialAveragePooling_updateOutput)(lua_State *L)
{
    THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
    int dW = luaT_getfieldcheckint(L, 1, "dW");
    int dH = luaT_getfieldcheckint(L, 1, "dH");
    int kW = luaT_getfieldcheckint(L, 1, "kW");
    int kH = luaT_getfieldcheckint(L, 1, "kH");
    int padW = luaT_getfieldcheckint(L, 1, "padW");
    int padH = luaT_getfieldcheckint(L, 1, "padH");


  real *output_data;
  real *input_data;

  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  long nbatch = 1;

  long inputWidth;
  long inputHeight;
  long outputWidth;
  long outputHeight;
  long nInputPlane; // number of channels (or colors)

  long k;

  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
    dimc++;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  nInputPlane = input->size[dimc];

  if(1)
  {
    outputWidth  = (long)(ceil((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
    outputHeight = (long)(ceil((float)(inputHeight - kH + 2*padH) / dH)) + 1;
  }
  else
  {
    outputWidth  = (long)(floor((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
    outputHeight = (long)(floor((float)(inputHeight - kH + 2*padH) / dH)) + 1;
  }
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  THArgCheck(inputWidth >= kW - 2 * padW && inputHeight >= kH - 2 * padH, 2, "input image smaller than kernel size");

  if (input->nDimension == 3)
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
  else
    THTensor_(resize4d)(output, input->size[0], nInputPlane, outputHeight, outputWidth);
  
  input = THTensor_(newContiguous)(input);
  THArgCheck(THTensor_(isContiguous)(output), 3, "output must be contiguous");
  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      long xx, yy;
      /* For all output pixels... */
      real *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
      real *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      long i;
      for(i = 0; i < outputWidth*outputHeight; i++)
        ptr_output[i] = 0;
      
      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          /* Compute the mean of the input image... */
          long hstart = yy * dH - padH;
          long wstart = xx * dW - padW;
          long hend = fminf(hstart + kH, inputHeight + padH);
          long wend = fminf(wstart + kW, inputWidth + padW);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = fmaxf(hstart, 0);
          wstart = fmaxf(wstart, 0);
          hend = fminf(hend, inputHeight);
          wend = fminf(wend, inputWidth);

          real sum = 0;

          int divide_factor;
          if(1)
            divide_factor = pool_size;
          else
            divide_factor = (hend - hstart) * (wend - wstart);

          long kx, ky;

          for(ky = hstart; ky < hend; ky++)
          {
            for(kx = wstart; kx < wend; kx++)
              sum += ptr_input[ky*inputWidth + kx];
          }
          /* Update output */
          *ptr_output++ += sum/divide_factor;
        }
      }
    }
  }
  return 1;
}



static const struct luaL_Reg nn_(SpatialAveragePooling__) [] = {
    {"SpatialAveragePooling_updateOutput", nn_(SpatialAveragePooling_updateOutput)},
    {NULL, NULL}
};


static void nn_(SpatialAveragePooling_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, nn_(SpatialAveragePooling__), "nn");
    lua_pop(L,1);
}

#endif
