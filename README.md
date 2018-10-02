# Introduction

`pyband` and `pydos` are two python scripts that take in the VASP calculation results (e.g. OUTCAR and PROCAR) and  convert the results to images. It offers a fast and effective way to preview the calcuated results. The image plotting utilizes `matplotlib` package.

# Examples
## `pyband`

When no arguments is given, `pyband` reads in `OUTCAR` (optionally `KPOINTS`) and find the band information within. It then plot the resulting band structure and save it as `band.png`.

```$ pyband```

![band_with_no_args](examples/band_no_args.png)

The default output image name  can be changed by adding `-o YourImageName.suffix` to the above command line.  Note that the image format is automatically recognized by the script, which can be any format that is supported by `matplotlib`. The size of the image can also be speified by `-s width height` command line arguments. 

In the above image, the name of the high-symmetry K-point is not shown in the resulting image, which can be specifiec by `-k` flag.

```$ pyband -k mgkm```

![band_with_kname](examples/band_with_kname.png)

In some cases, if you are interested in finding out the characters of each KS states, e.g. the contribution of some atom to each KS state, the flag `--occ atoms` comes into help.

```$ pyband --occ '1 3 4'```

![band_with_atom_weight](examples/band_with_atoms_weight.png)
where in the above image, `1 3 4` are the atom index starting from 1 to #atoms. The size of red dots in the figure indicates the weight of the specified atoms to the KS states.  This can also be represented using a colormap:

```$ pyband --occ '1 3 4' --occL```

![band_with_atom_weight_cmap](examples/band_with_atoms_weight_cmap.png)

Or if you are interested in the 