# Monte Carlo Method to Estimate $\pi$ on CUDA

## Running CUDA C Code on Google Colab [^1]
If you want to work on [Google Colab](https://colab.google/), you should follow the steps below: 
* Create a **new notebook** on Google Colab
* Click on the **runtime** button from the menu above and click on **Change runtime type** from the window that opens
* Select **T4 GPU** from the pop-up that opens and save
* Click the **+Code** button in the top left and add a **code cell**.
* Paste the code snippet below and click the **run cell** button or press **CTRL+Enter**

```python
# check python version to verify automatically installed
!python --version

# check nvcc (CUDA compiler driver) version to verify the automatically installed
!nvcc --version

# since google colab runs jupyter notebook, we need to install nvcc4jupyter: cuda c++ plugin for jupyter notebook
!pip install nvcc4jupyter

# after installing, load the package (or extension)
%load_ext nvcc4jupyter
```

> Click [here](https://github.com/andreinechaev/nvcc4jupyter) to learn more about the `nvcc4jupyter` plugin.

To see that we have successfully installed CUDA, let's add a new code cell and run the test code below.

```python
%%cuda
#include <stdio.h>
__global__ void hello(){
 printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}
int main(){
 hello<<<2, 2>>>();
 cudaDeviceSynchronize();
}
```
---

## What is [Monte Carlo Method]([url](https://en.wikipedia.org/wiki/Monte_Carlo_method))? 
Monte Carlo methods, or Monte Carlo experiments, are a broad class of computational algorithms that rely on repeated random sampling to obtain numerical results. The underlying concept is to use randomness to solve problems that might be deterministic in principle. The name comes from the Monte Carlo Casino in Monaco, where the primary developer of the method, mathematician Stanislaw Ulam, was inspired by his uncle's gambling habits. (Wikipedia)

This is where we can take advantage of how quickly a computer can generate pseudorandom numbers. There is a whole class of algorithms called Monte  Carlo simulations that exploit randomness to estimate real-world scenarios that would otherwise be difficult to calculate explicitly. We can use a  Monte Carlo simulation to  estimate the area ratio of the circle to the square.[^2]

Imagine we randomly hit darts into the area of the square. We get this estimate by counting the total number of darts in the square (all of them since we always hit the square) to the total number of darts inside the circle. Multiply the estimated ratio by four and we get an estimate for $\pi$. The more dart we use the more accurate our estimate of $\pi$.  

---

## Mathematical Explanation of This Problem
For the sake of simplicity of mathematical operations, let us consider a concentric circle of length 1 feet and a square inscribing this circle. 

<p align="center">
  <img src="https://i.ibb.co/1RBMKT9/Screenshot-from-2024-10-12-12-14-52.png?raw=true" width="200px" height="200px" alt="concentric unit circle and square"/>
</p>

$$ \text{Area Circle}: A_c = \pi.r^2 $$
$$ \text{Area Square}: A_s = (2r)^2 = 4r^2 $$
$$ \text{The ratio of the two areas is}: \frac{A_c}{A_s} = \frac{\pi.\bcancel{r^2}}{4\bcancel{r^2}} $$
$$ \text{Let's solve for pi}: \pi = \frac{4A_c}{A_s} \hspace{1.5cm} \text{(1)}$$ 

If we have an estimate for the ratio of the area of the circle to the area of the square we can solve for $\pi$. The challenge becomes estimating this ratio.  

This ratio can be interpreted probabilistically: if we randomly toss darts uniformly into the square, the proportion of darts that land inside the circle (compared to the total number of darts) should be approximately equal to the ratio of the area of the circle to the area of the square.

If we toss $N$ random darts, the number of darts that land inside the circle, say ${N_\text{circle}}$​, will approximately satisfy:

$$ \frac{N_\text{circle}}{N}\approx \frac{4 \times A_c}{A_s} = \pi \hspace{1.5cm} \text{(2)}$$

---

## Algorithm for NON-Parallel Version [^3]
1. Define the variable `number_of_tosses` (referring to $N$ in eq. 2), and specify how many iterations we will estimate $\pi$. Remember, every toss has to fall inside the square!
2. Define the variable `toss` and assign it 0 to use every iteration.
3. Define the variable `number_in_circle` (referring to $N_\text{circle}$ in eq. 2) and assign it 0. We will use this variable for tosses that fall inside the circle.
4. Since we are working in **two-dimensional space**, randomly generate $x$ and $y$ value ​​between $-1$ and $1$.
5. If the coordinates lie inside a circle, then they must satisfy the equation $x^2 + y^2 \leq 1$, and if so, this point lies not only inside the square but also inside the circle. Therefore `number_in_circle` must be incremented by one.
6. Increment `toss` by one and repeat from **2** until **`toss` $=$ `number_of_tosses`.**
7. Calculate $\pi$ using eq. 2.
8. Print estimated $\pi$ value.

---





[^1]: https://medium.com/@zubair09/running-cuda-on-google-colab-d8992b12f767
[^2]: https://courses.cs.washington.edu/courses/cse160/16wi/sections/07/Section07Handout.pdf
[^3]: https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/
