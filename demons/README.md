# pydemons
A proof-of-concept Python implementation of the (symmetric) log-domain diffeormorphic demons algorithm

Most of the code was inspired by [Herve Lambaert's Matlab implementation](http://www.mathworks.com/matlabcentral/fileexchange/39194-diffeomorphic-log-demons-image-registration).

Code majorly copied from [dohmatob](https://github.com/dohmatob/pydemons)

Example
=======
To demo the code, run the following command (in a terminal)

    python plot_demons.py -fixed=<path_to_image1> -moving=<path_to_image2>

To replicate 19th May, 2021 slides results
```
python plot_demons.py -fixed=./data/autocropped_unit10.jpg -moving=./data/man_cropped_unit11.jpg
```
