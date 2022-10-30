算法hw1

  Consider the algorithm: run the L-V algorithm for 2T(n) steps, and if it has not stopped yet, just abort it. Trivially, the running time is now

2T(n)

 Note that we can analyze our Las Vegas algorithm to obtain its expected running time on input ![x](https://s0.wp.com/latex.php?latex=x&bg=ffffff&fg=000&s=0&c=20201002), say ![p(x)](https://s0.wp.com/latex.php?latex=p%28x%29&bg=ffffff&fg=000&s=0&c=20201002). We know that our Monte Carlo algorithm will output a correct answer at least when the simulation finishes in ![t(x)](https://s0.wp.com/latex.php?latex=t%28x%29&bg=ffffff&fg=000&s=0&c=20201002) steps. If ![X](https://s0.wp.com/latex.php?latex=X&bg=ffffff&fg=000&s=0&c=20201002) is the running time of the simulation of our Las Vegas algorithm, then the probability of having a correct answer is at least

![\text{Pr}[X < t(x)] = 1- \text{Pr} [ X \ge t(x) ]](https://s0.wp.com/latex.php?latex=%5Ctext%7BPr%7D%5BX+%3C+t%28x%29%5D+%3D+1-+%5Ctext%7BPr%7D+%5B+X+%5Cge+t%28x%29+%5D&bg=ffffff&fg=000&s=0&c=20201002)
![\ge 1 - \frac{ \mathbb{E}[X] }{t(x)}](https://s0.wp.com/latex.php?latex=%5Cge+1+-+%5Cfrac%7B+%5Cmathbb%7BE%7D%5BX%5D+%7D%7Bt%28x%29%7D&bg=ffffff&fg=000&s=0&c=20201002)
![= 1- \frac{p(x)}{3p(x)}](https://s0.wp.com/latex.php?latex=%3D+1-+%5Cfrac%7Bp%28x%29%7D%7B3p%28x%29%7D&bg=ffffff&fg=000&s=0&c=20201002)
![= \frac23,](https://s0.wp.com/latex.php?latex=%3D+%5Cfrac23%2C&bg=ffffff&fg=000&s=0&c=20201002)

which is good, if we pick ![t(x)=3p(x)](https://s0.wp.com/latex.php?latex=t%28x%29%3D3p%28x%29&bg=ffffff&fg=000&s=0&c=20201002). Obviously, we can also pick larger ![t(x)](https://s0.wp.com/latex.php?latex=t%28x%29&bg=ffffff&fg=000&s=0&c=20201002) to gain better probability. But, as long as we pick ![t(x)>2p(x)](https://s0.wp.com/latex.php?latex=t%28x%29%3E2p%28x%29&bg=ffffff&fg=000&s=0&c=20201002), we can also amplify this probability by doing multiple simulations and process a final answer (in case of Boolean answer, for example, we do majority voting).

![image-20221001171205711](C:\Users\12636\AppData\Roaming\Typora\typora-user-images\image-20221001171205711.png)

![image-20221001171226623](C:\Users\12636\AppData\Roaming\Typora\typora-user-images\image-20221001171226623.png)

![image-20221001171248723](C:\Users\12636\AppData\Roaming\Typora\typora-user-images\image-20221001171248723.png)