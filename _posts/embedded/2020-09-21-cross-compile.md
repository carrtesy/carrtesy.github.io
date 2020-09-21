---
title: "Embedded Systems: Cross Compilation"

date: 2020-09-21

last_modified_at: 2020-09-21

categories:
 - Embedded System
 - Raspberry Pi

Using host machine to compile codes that is active on target machine.

### Setup
- Host machine
	- OS: ubuntu 18.04 native
- Target machine
	- raspberry pi 3b+
	
### Getting Toolchain
```
	sudo apt install gcc-arm-linux-gnueabihf
```

This command installs **gcc** working on:
- arm processor
- EABI is hard (gcc -mfloat-abi=hard)

### Crosstool-NG
download
```
wget http://crosstool-ng.org/download/crosstool-ng/crosstool-ng-1.24.0.tar.xz
```

decompress by
```
tar -xvf crosstool-ng-1.24.0.tar.xz
```

for required packages
```
sudo apt install autoconfflex bison texinfohelp2man gawk libtool-bin
```

curses
```
sudo apt-get install libncurses5-dev libncursesw5-dev
```

configure by
```
./configure
```
in decompressed dir.

make by:
```
make
sudo make install
```


### check configurations
list up
```
ct-ng list-samples
```	

check specific configuration
```
ct-ng arm-unknown-linux-gnueabi
```
	
adjust configuration by
```
ct-ng menuconfig
```

build
```
ct-ng build
```