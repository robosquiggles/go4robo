# `utils` Notes

## `xacro.py`

Used to convert any `.urdf.xacro` file to a `.urdf` file without ROS. I did this to create it:

### Use
Do this to convert a `.urdf.xacro` file to a `.urdf` file.
```
cp xacro.py [directory containing /urdf]
python3 [directory containing /urdf]/xacro.py -o [output file path] [input file path]
rm [directory containing /urdf]/xacro.py
```

### Where it came from
[https://github.com/doctorsrn/xacro2urdf?tab=readme-ov-file](https://github.com/doctorsrn/xacro2urdf?tab=readme-ov-file)