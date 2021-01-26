#!/bin/bash
SRC='libco'
git clone https://github.com/SergioMartin86/libco.git ${SRC}
(cd ${SRC} && make MPICXX=${CC})

exit 0
