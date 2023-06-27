#!/bin/bash
mkdir Seq_003
ffmpeg -i Seq_003.mov -framerate 40 -s 342x256 Seq_003/%d.png
mkdir Seq_011
ffmpeg -i Seq_011.mov -framerate 40 -s 342x256 Seq_011/%d.png
mkdir Seq_013
ffmpeg -i Seq_013.mov -framerate 40 -s 342x256 Seq_013/%d.png
mkdir Seq_093
ffmpeg -i Seq_093.mov -framerate 40 -s 342x256 Seq_093/%d.png
mkdir Seq_094
ffmpeg -i Seq_094.mov -framerate 40 -s 342x256 Seq_094/%d.png