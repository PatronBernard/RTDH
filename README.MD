# RTDH-CUDA-GLFW
Real-Time Digital Holography accelerated with CUDA and visualized with GLFW

<h4>Required libraries</h4>
<p>I listed the versions that I created the project with, other versions are untested but will likely work.</p>
<ul>
<li>Vimba 2.0</li>
<li>CUDA 6.5 </li>
<li>GLEW 1.13.0</li>
<li>GLFW 3.1.2 (WIN64)</li>
<li>ImGui 1.48</li>
<li>(Visual Studio 2013 or another CUDA-compatible VS version)</li>
</ul>
<h4>Launching</h4>
The program (RTDH.exe) will first search for a camera. If it does not find one, it will exit and report the reason in log.txt. 

<h4>Configuring the camera</h4>
Next it will load the CameraSettings.xml file onto the camera. Changing the resolution seems a bit quirky, sometimes the settings stick, sometimes they don't. You can also use the Vimba viewer to configure the camera in real time and save the .xml file (untested). The exposure time is also changed in this file.

<h4>Using the different modes</h4>
The camera has the following 4 modes:
<ul>
<li>Video mode: displays the raw image that the CCD registers. Useful for alignment or other optical experiments.</li>
<li>Reconstruction mode: real-time Fresnel reconstruction. You can adjust the reconstruction distance with the slider. You can also display the phase, although it is meaningless here because there is no phase to compare it to.</li>
<li>Holographic interferometry mode: displays the phase difference relative to a recorded phase (which is achieved by pressing R).</li>
<li>Fourier Transform mode: plain Fourier transform of the CCD image. In conjuction with a lens you could draw periodic patterns on a paper, film it and see the 2D power spectrum.</li>
<li>View recorded frame mode: speaks for itself.</li>
</ul>
Note that upon exiting the application, the recorded reconstructed hologram will be exported to a binary file (reconstructed_hologram.bin). 

<h4>Hotkeys</h4>
<b>Esc</b> - Close application

<b>1</b> - Select video mode

<b>2</b> - Fresnel reconstruction

<b>3</b> - Holographic interferometry

<b>R</b> - Record a frame for holographic interferometry

<b>V</b> - View recorded frame

<b>4</b> - FFT

<b>M</b> - Display magnitude (if applicable)

<b>P</b> - Display phase (if applicable)

<b>C</b> - Show/hide control dialog

For additional support you can contact me at jan.morez [AT] gmail [DOT] com.
