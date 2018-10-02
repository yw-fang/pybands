# Introduction

`pyband` and `pydos` are two python scripts that analyse the VASP calculation
results (e.g. OUTCAR and PROCAR) and  convert the results to images. It offers a
fast and effective way to preview the calcuated results. The image plotting
utilizes `matplotlib` package.

# Examples
## pyband

When no argument is given, `pyband` reads in `OUTCAR` (optionally `KPOINTS`)
and find the band information within. It then plots the resulting band structure
and save it as `band.png`.

```$ pyband```

![band_with_no_args](examples/band_no_args.png)

The default output image name  can be changed by adding `-o
YourImageName.suffix` to the above command line.  Note that the image format is
automatically recognized by the script, which can be any format that is
supported by `matplotlib`. The size of the image can also be speified by `-s
width height` command line arguments. 

The labels of the high-symmetry K-points, which are not shown in the figure, can
be designate by `-k` flag.

```$ pyband -k mgkm```

![band_with_kname](examples/band_with_kname.png)

In some cases, if you are interested in finding out the characters of each KS
states, e.g. the contribution of some atom to each KS state, the flag `--occ
atoms` comes to help.

```$ pyband --occ '1 3 4'```

![band_with_atom_weight](examples/band_with_atoms_weight.png)

where `1 3 4` are the atom index starting from 1 to #atoms  in the above image.
The size of red dots in the figure indicates the weight of the specified atoms
to the KS states.  This can also be represented using a colormap:

```$ pyband --occ '1 3 4' --occL```

![band_with_atom_weight_cmap](examples/band_with_atoms_weight_cmap.png)

The spd-projected weight can also be specefied:

```$ pyband --occ '1 3 4' --spd '4 5 6 7 8' ```

![band_with_atom_weight_spd](examples/band_with_atoms_weight_spd.png)

where in the arguments of `--spd`:


> s orbital: 0

> py, pz, px orbital: 1 2 3

> dxy, dyz, dz2, dxz, dx2 orbital: 4 5 6 7 8

More command line arguments can be found by `pyband -h`.

## pydos

This script is used to plot partial density of states (pDOS) from VASP `PROCAR`
files. 

`pydos -p '1 3 4' -p '2 7 8' -p '5 6 9' -z 0.65 -x -1 2  -y 0 6`

![pdos_example](examples/dos_p3.png)

where `-p` specifies the atom indexes, `-x` and `-y` determines the x and y
limits of the plot, `-z` is followed by the energy reference of the plot.


## npdos

This script can plot PDOS from multiple VASP `PROCAR`s in multiple axes, example usages:

```bash

#!/bin/bash

npdos -nr 2 -f 4.8 4.0 \
    -o g1.png \
    -nxminor 4 \ 
    -i pbe/scf/PROCAR         -a 0 -p 0 -pv n -tlab 'PBE-PBE' -tlw 0.5 -tlc r \ 
    -i scf-pbe_opt-hse/PROCAR -a 0 -p 0 -pv n -tlab 'HSE-PBE' -tlw 0.5 -tlc b \ 
    -i scf-hse_opt-pbe/PROCAR -a 1 -p 0 -pv n -tlab 'PBE-HSE' -tlw 0.5 -tlc r \ 
    -i hse/scf/PROCAR         -a 1 -p 0 -pv n -tlab 'HSE-HSE' -tlw 0.5 -tlc b \ 
    -x -4 6 -x -6 6 \ 
    -z 3.3129 -z 3.3726 -z 3.5583 -z 3.6332 \
    -panelloc 0.01 0.95 \
    -q  
```

The resulting figure:

![npdos_example](examples/dos_p5.png)
