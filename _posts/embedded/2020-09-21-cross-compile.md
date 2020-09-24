---
title: "Embedded Systems: Cross Compilation"

date: 2020-09-21

last_modified_at: 2020-09-21

categories:
 - Embedded System
 - Raspberry Pi
---

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

### hello world
After finishing the process above,
toolchain is located in:
```
~/x-tools/arm-unknown-linux-gnueabi/bin
```

so setup the path by:
```
export PATH=$PATH:~/x-tools/arm-unknown-gnueabi/bin
```

and can create hello.c as:
```c
#include <stdio.h>

int main(void){
	int value;
	printf("Hello Raspberry Pi!");
	printf("type an integer value:");
	scanf("%d", &value);
	printf("Input value: %d %x \n", value, value);
	return 0;
}
```

and compile by
```
arm-unknown-linux-gnueabi-gcc hello.c -o hello
```

However, this cannot be executed, as we are on host machine, not on target machine.

### Connect to target machine(Raspberry Pi)

I used ethernet cable / adapter to connect host and target machine.
Hope this helps!


![host_nw_setup](/assets/images/embedded/host_nw_setup.png  "host_nw_setup")

Network
for host machine: 192.168.1.1
for target machine: 192.168.1.10


