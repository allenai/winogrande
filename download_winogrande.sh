#!/bin/sh

wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1_beta.zip
unzip winogrande_1.1_beta.zip; rm -rf __MACOSX
mv winogrande_1.1_beta data
rm winogrande_1.1_beta.zip

