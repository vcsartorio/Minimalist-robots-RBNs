all: bootldr CRWLEVY Target dritto ohc ohc-arduino-8mhz ohc-arduino-16mhz test-for

.PHONY: docs bootldr CRWLEVY Target dritto ProvaStampa ohc ohc-arduino-8mhz ohc-arduino-16mhz test-for
KILOLIB = build_local/kilolib.a

INC = -I. #As the behaiors are in the behavior folder and not this one, we have to include this folder to the include path
BEHAVIOR_SOURCE_FOLDER = ../behaviors
BEHAVIOR_BUILD_FOLDER = ../../build/behaviors_real
REAL = 1 # To include debug in real experiment 

bootldr: build_local/bootldr.elf build_local/bootldr.hex build_local/bootldr.lss
# CRWLEVY: $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY.elf $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY.hex $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY.lss
Target: $(BEHAVIOR_BUILD_FOLDER)/Target.elf $(BEHAVIOR_BUILD_FOLDER)/Target.hex $(BEHAVIOR_BUILD_FOLDER)/Target.lss
dritto: $(BEHAVIOR_BUILD_FOLDER)/dritto.elf $(BEHAVIOR_BUILD_FOLDER)/dritto.hex $(BEHAVIOR_BUILD_FOLDER)/dritto.lss

#ProvaStampa: build/ProvaStampa.elf build/ProvaStampa.hex build/ProvaStampa.lss


CC = avr-gcc
AVRAR = avr-ar
AVROC = avr-objcopy
AVROD = avr-objdump
AVRUP = avrdude
# C_DEFINES = -DCRW_EXPONENT=0.30 -DLEVY_EXPONENT=1.0 -DREAL=1

PFLAGS = -P usb -c avrispmkII # user to reprogram OHC
CFLAGS = -mmcu=atmega328p -Wall -gdwarf-2 -O3 -funsigned-char -funsigned-bitfields -fpack-struct -fshort-enums 
CFLAGS += -DF_CPU=8000000
# CFLAGS += $(C_DEFINES)
ASFLAGS = $(CFLAGS)
BOOTLDR_FLAGS = -Wl,-section-start=.text=0x7000 -DBOOTLOADER
# FLASH = -R .eeprom -R .fuse -R .lock -R .signature
EEPROM = -j .eeprom --set-section-flags=.eeprom="alloc,load" --change-section-lma .eeprom=0  

%.lss: %.elf
	$(AVROD) -d -S $< > $@

%.hex: %.elf
	$(AVROC) -O ihex $(FLASH) $< $@

%.eep: %.elf
	$(AVROC) -O ihex $(EEPROM) $< $@

%.bin: %.elf
	$(AVROC) -O binary $(FLASH) $< $@ 

build_local:
	mkdir -p $@

build_real:
	mkdir -p $(BEHAVIOR_BUILD_FOLDER)

$(KILOLIB): kilolib.o message_crc.o message_send.o | build_local
	$(AVRAR) rcs $@ kilolib.o message_crc.o message_send.o 
	rm -f *.o

build_local/bootldr.elf: bootldr.c kilolib.c message_crc.c | build_local
	$(CC) $(CFLAGS) $(BOOTLDR_FLAGS) -o $@ bootldr.c kilolib.c message_crc.c

# $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY.elf: $(BEHAVIOR_SOURCE_FOLDER)/CRWLEVY.c $(KILOLIB) | build_real
# 	$(CC) $(INC) $(CFLAGS) $(C_DEFINES) -o $@ $< $(KILOLIB) 

$(BEHAVIOR_BUILD_FOLDER)/Target.elf: $(BEHAVIOR_SOURCE_FOLDER)/Target.c $(KILOLIB) | build_real
	$(CC) $(INC) $(CFLAGS)  -o $@ $< $(KILOLIB) 

$(BEHAVIOR_BUILD_FOLDER)/dritto.elf: $(BEHAVIOR_SOURCE_FOLDER)/dritto.c $(KILOLIB) | build_real
	$(CC) $(INC) $(CFLAGS)  -o $@ $< $(KILOLIB)

#
#build/ProvaStampa.elf: ProvaStampa.c $(KILOLIB) | build
#	$(CC) $(CFLAGS) -o $@ $< $(KILOLIB) 


program-boot: build_local/bootldr.hex
	$(AVRUP) -p m328p $(PFLAGS) -U "flash:w:$<:i"

program-CRWLEVY: $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY_2.0_0.90.hex
	$(AVRUP) -p m328p $(PFLAGS) -U "flash:w:$<:i" -U "flash:w:build_local/bootldr.hex" -D

program-Target: $(BEHAVIOR_BUILD_FOLDER)/Target.hex
	$(AVRUP) -p m328p $(PFLAGS) -U "flash:w:$<:i" -U "flash:w:build_local/bootldr.hex"

program-dritto: $(BEHAVIOR_BUILD_FOLDER)/dritto.hex
	$(AVRUP) -p m328p $(PFLAGS) -U "flash:w:$<:i" -U "flash:w:build_local/bootldr.hex"

#program-Provastampa: build/ProvaStampa.hex
#	$(AVRUP) -p m328p $(PFLAGS) -U "flash:w:build/ProvaStampa.hex:i" -U "flash:w:build/bootldr.hex"


docs:
	cat message.h kilolib.h message_crc.h | grep -v "^\#" > docs/kilolib.h
	(cd docs; doxygen)

clean:
	rm -fR build


# test-for: $(BEHAVIOR_SOURCE_FOLDER)/simulation_rbn.c $(KILOLIB) | build_real
# 	for alpha in 1.0 1.2 1.4 1.6 1.8 2.0 ; do \
# 		for rho in 0.00 0.15 0.30 0.45 0.60 0.75 0.90 0.95 0.99 1.0; do \
# 			export C_DEFINES_CUSTOM="-DCRW_EXPONENT=$$rho -DLEVY_EXPONENT=$$alpha -DREAL=1 -DMAX_TIME=18600";\
# 			$(CC) $(INC) $(CFLAGS) $$C_DEFINES_CUSTOM -o $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY_$${alpha}_$${rho}.elf $< $(KILOLIB);\
# 			$(AVROD) -d -S $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY_$${alpha}_$${rho}.elf > $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY_$${alpha}_$${rho}.lss;\
# 			$(AVROC) -O ihex $(FLASH) $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY_$${alpha}_$${rho}.elf $(BEHAVIOR_BUILD_FOLDER)/CRWLEVY_$${alpha}_$${rho}.hex;\
# 		done \
# 	done
