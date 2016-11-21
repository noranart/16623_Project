#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxPooling.c"
#else

static void nn_(SpatialMaxPooling_updateOutput_frame)(          real *input_p,
                                                      real *output_p,
                                                      real *ind_p,
                                                      long nslices,
                                                      long iwidth,
                                                      long iheight,
                                                      long owidth,
                                                      long oheight,
                                                      int kW,
                                                      int kH,
                                                      int dW,
                                                      int dH,
                                                      int padW,
                                                      int padH,
                                                      int dilationW,
                                                      int dilationH)

{
    long k;
    for (k = 0; k < nslices; k++)
    {
        /* loop over output */
        long i, j;
        real *ip = input_p   + k*iwidth*iheight;
        for(i = 0; i < oheight; i++)
        {
            for(j = 0; j < owidth; j++)
            {
                long hstart = i * dH - padH;
                long wstart = j * dW - padW;
                long hend = fminf(hstart + (kH - 1) * dilationH + 1, iheight);
                long wend = fminf(wstart + (kW - 1) * dilationW + 1, iwidth);
                while(hstart < 0)
                    hstart += dilationH;
                while(wstart < 0)
                    wstart += dilationW;
                
                /* local pointers */
                real *op = output_p  + k*owidth*oheight + i*owidth + j;
                real *indp = ind_p   + k*owidth*oheight + i*owidth + j;
                
                /* compute local max: */
                long maxindex = -1;
                real maxval = -THInf;
                long tcntr = 0;
                long x,y;
                for(y = hstart; y < hend; y += dilationH)
                {
                    for(x = wstart; x < wend; x += dilationW)
                    {
                        tcntr = y*iwidth + x;
                        real val = *(ip + tcntr);
                        if (val > maxval)
                        {
                            maxval = val;
                            maxindex = tcntr;
                        }
                    }
                }
                
                /* set output to local max */
                *op = maxval;
                
                /* store location of max */
                *indp = maxindex;
            }
        }
    }
}


static int nn_(SpatialMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
    int padW = luaT_getfieldcheckint(L, 1, "padW");
    int padH = luaT_getfieldcheckint(L, 1, "padH");
    int dilationW = 1;
    int dilationH = 1;

  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

        int dimw = 2;
        int dimh = 1;
        long nbatch = 1;
        long nslices;
        long iheight;
        long iwidth;
        long oheight;
        long owidth;
        real *input_data;
        real *output_data;
        real *indices_data;
        
        
        THArgCheck(input->nDimension == 3 || input->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");
        
        if (input->nDimension == 4)
        {
            nbatch = input->size[0];
            dimw++;
            dimh++;
        }
        THArgCheck(input->size[dimw] >= kW - padW && input->size[dimh] >= kH - padH, 2, "input image smaller than kernel size");
        THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");
        
        /* sizes */
        nslices = input->size[dimh-1];
        iheight = input->size[dimh];
        iwidth = input->size[dimw];
        if (1)
        {
            oheight = (long)(ceil((float)(iheight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
            owidth  = (long)(ceil((float)(iwidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
        }
        else
        {
            oheight = (long)(floor((float)(iheight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
            owidth  = (long)(floor((float)(iwidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
        }
        
        if (owidth < 1 || oheight < 1)
            THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
                    nslices,iheight,iwidth,nslices,oheight,owidth);
        
        if (padW || padH)
        {
            // ensure that the last pooling starts inside the image
            if ((oheight - 1)*dH >= iheight + padH)
                --oheight;
            if ((owidth  - 1)*dW >= iwidth  + padW)
                --owidth;
        }
        
        /* get contiguous input */
        input = THTensor_(newContiguous)(input);
        
        /* resize output */
        if (input->nDimension == 3)
        {
            THTensor_(resize3d)(output, nslices, oheight, owidth);
            /* indices will contain the locations for each output point */
            THTensor_(resize3d)(indices,  nslices, oheight, owidth);
            
            input_data = THTensor_(data)(input);
            output_data = THTensor_(data)(output);
            indices_data = THTensor_(data)(indices);
            
            nn_(SpatialMaxPooling_updateOutput_frame)(input_data, output_data,
                                                        indices_data,
                                                        nslices,
                                                        iwidth, iheight,
                                                        owidth, oheight,
                                                        kW, kH, dW, dH,
                                                        padW, padH,
                                                        dilationW, dilationH
                                                        );
        }
        else
        {
            long p;
            
            THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
            /* indices will contain the locations for each output point */
            THTensor_(resize4d)(indices, nbatch, nslices, oheight, owidth);
            
            input_data = THTensor_(data)(input);
            output_data = THTensor_(data)(output);
            indices_data = THTensor_(data)(indices);
            
            for (p = 0; p < nbatch; p++)
            {
                nn_(SpatialMaxPooling_updateOutput_frame)(input_data+p*nslices*iwidth*iheight, output_data+p*nslices*owidth*oheight,
                                                            indices_data+p*nslices*owidth*oheight,
                                                            nslices,
                                                            iwidth, iheight,
                                                            owidth, oheight,
                                                            kW, kH, dW, dH,
                                                            padW, padH,
                                                            dilationW, dilationH
                                                            );
            }
        }
        
        /* cleanup */
    return 1;
}

    
    
    
    
static const struct luaL_Reg nn_(SpatialMaxPooling__) [] = {
  {"SpatialMaxPooling_updateOutput", nn_(SpatialMaxPooling_updateOutput)},
  {NULL, NULL}
};

static void nn_(SpatialMaxPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
