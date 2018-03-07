function avg = multi_scale_detail_manipulation(im)
    rgb = double(im)/65535;
    cform = makecform('srgb2lab');
    lab = applycform(rgb, cform);
    L = lab(:,:,1);

    %% Filter
    L0 = wlsFilter(L, 0.125, 1.2);
    L1 = wlsFilter(L, 0.50,  1.2);

    %% Fine
    val0 = 25;
    val1 = 1;
    val2 = 1;
    exposure = 1.0;
    saturation = 1.1;
    gamma = 1.0;

    fine = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    %% Medium
    val0 = 1;
    val1 = 40;
    val2 = 1;
    exposure = 1.0;
    saturation = 1.1;
    gamma = 1.0;

    med = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    %% Coarse
    val0 = 4;
    val1 = 1;
    val2 = 15;
    exposure = 1.10;
    saturation = 1.1;
    gamma = 1.0;

    coarse = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    %% Avg
    avg = (fine + med + coarse)/3;
end