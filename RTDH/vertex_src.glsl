#version 150
#define PI 3.14159265358979323846264338327950
//Found on https://www.shadertoy.com/view/4dXXDX

uniform mat4 Projection; 

in vec2 in_Position;
in float in_Magnitude;
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

//Hot colormap, black to red-yellow-white
vec3 green(float t)
{
    return vec3(smoothstep(0.50,1.00,t),
                smoothstep(0.00,0.50,t),
                smoothstep(0.50,1.00,t));
}

vec3 bw(float t)
{
	return vec3(clamp(t,0.0,1.0)/3.0,
				clamp(t,0.0,1.0)/3.0,
				clamp(t,0.0,1.0)/3.0);
}

float colormap_red(float x) {
	if (x < 0.7) {
		return 4.0 * x - 1.5;
	}
	else {
		return -4.0 * x + 4.5;
	}
}

float colormap_green(float x) {
	if (x < 0.5) {
		return 4.0 * x - 0.5;
	}
	else {
		return -4.0 * x + 3.5;
	}
}

float colormap_blue(float x) {
	if (x < 0.3) {
		return 4.0 * x + 0.5;
	}
	else {
		return -4.0 * x + 2.5;
	}
}

vec3 jet(float x) {
	float r = clamp(colormap_red(x), 0.0, 1.0);
	float g = clamp(colormap_green(x), 0.0, 1.0);
	float b = clamp(colormap_blue(x), 0.0, 1.0);
	return vec3(r, g, b);
}

void main(void){
	gl_Position=vec4(in_Position.x, in_Position.y,0.0f,1.0f);
	//ex_Color = green(in_Magnitude);
	ex_Color = hot(in_Magnitude);
	//ex_Color = jet(in_Magnitude);
	//ex_Color = wheel(in_Magnitude);
	//ex_Color = bw(in_Magnitude);
}