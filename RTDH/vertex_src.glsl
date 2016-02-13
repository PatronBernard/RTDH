#version 150
#define PI 3.14159265358979323846264338327950
//Found on https://www.shadertoy.com/view/4dXXDX
in  vec2 in_Position;
in vec2 in_Complex;
//We should add a vec2 that contains the real and imaginary parts. 
out vec3 ex_Color;

// Jet colormap
vec3 wheel(float t)
{
    return clamp(abs(fract(t + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) -1.0, 0.0, 1.0);
}

//Hot colormap, black to red-yellow-white
vec3 hot(float t)
{
    return vec3(smoothstep(0.00,0.33,t),
                smoothstep(0.33,0.66,t),
                smoothstep(0.66,1.00,t));
}

void main(void){
    //Map height to a color, will change into atan(b/a) to map phase to colour
    //ex_Color = hot(0.5*in_Position.z+0.5);
	//Dirty solution to z-values that can exceed 1
	gl_Position=vec4(in_Position.x,in_Position.y,0,1);
	//ex_Color=hot(in_Position.x);
	ex_Color = hot(in_Complex.x);
}