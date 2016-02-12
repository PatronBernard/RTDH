# RTDH-CUDA-GLFW
Real-Time Digital Holography accelerated with CUDA and visualized with GLFW

<h4>Required libraries</h4>
<p>I listed the versions that I created the project with, other versions are untested but will likely work.</p>
<ul>
<li>CUDA 7.5</li>
<li>GLEW 1.13.0</li>
<li>GLFW 3.1.2 (WIN64)</li>
<li>(Visual Studio Express 2013 or another CUDA-compatible VS version)</li>
</ul>
<p> It is critical that you use the x64 binaries for GLFW, or you'll get unresolved external symbol-errors.</p>
<h4>Configuring Visual Studio </h4>
<p>I've noticed that simply cloning this project and opening the .sln file causes lots of trouble, partially because the user might have their library directories in a different location or the user has a different VS version. Here are the basic steps that should allow you to build the project:</p>
<b>1.</b> Create a new VS-project CUDA 7.5 Runtime project (only possible if you selected the "Visual Studio Integration" option during the CUDA-installation). This will save you a lot of work, doing this manually is possible but tedious. <br>
<b>2.</b> Add all the source files (.cpp, .h and .cu files) <br>
<b>3.</b> In the <i>VC++ Directories</i> tab, add the  <i>GLEW/include</i> and <i>GLFW/include</i> directories. <br>
<b>4.</b> In the <i>Linker>General</i> tab, add the <i>GLEW/lib</i> and GLFW/lib</i> directories. Make sure that you use the VS-2013 libraries, as the toolset of the project is v120. <br>
<b>5.</b> In the <i>Linker>Input tab</i>, add the following files: <br>
<ul>
<li>glfw3.lib</li>
<li>Release\Win32\glew32s.lib</li>
<li>opengl32.lib</li>
</ul>