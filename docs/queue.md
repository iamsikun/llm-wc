# M/M/s Queue

Consider a queueing system with Poisson arrival with rate $\lambda$ and $s$ servers. Each server has the same service rate $\mu$.

## Steady State Equations

When the number of customers in the system, denoted as $k\in \{0, 1, \dots, \}$, is smaller than $s-1$, the flow balance in steady state is:
$$
p_{k-1} \lambda + p_{k+1} (k+1)\mu = p_k (\lambda + k\mu),\;\forall\; k\in \{0, \dots, s-1\}.
$$
When the number of customers is at least the number of servers, the flow balance is:
$$
p_{k-1} \lambda + p_{k+1} s\mu = p_k (\lambda + s\mu),\;\forall\; k\in \{s, s+1, \dots, \}.
$$

Together, we have:
$$
\begin{aligned}
& \mu p_1 = \lambda p_0, & \Rightarrow p_1 = \frac{\lambda}{\mu} p_0\\
& 2\mu p_2 + \lambda p_0 = (\lambda + \mu) p_1 & \Rightarrow p_2 = \frac{\lambda}{2\mu} p_1\\
& 3\mu p_3 + \lambda p_1 = (\lambda + 2\mu) p_2 & \Rightarrow p_3 = \frac{\lambda}{3\mu} p_2\\
& \vdots \\
& s\mu p_s + \lambda p_{s-2} = [\lambda + (s-1)\mu] p_{s-1} & \Rightarrow p_{s} = \frac{\lambda}{s\mu} p_{s-1}\\
& s\mu p_{s+1} + \lambda p_{s-1} = (\lambda + s\mu) p_s & \Rightarrow p_{s+1} = \frac{\lambda}{s\mu} p_{s}\\
& s\mu p_{s+2} + \lambda p_s = (\lambda + s\mu) p_{s+1} & \Rightarrow p_{s+2} = \frac{\lambda}{s\mu} p_{s+1}\\
& \vdots \\
\end{aligned}
$$

Let $a = \lambda/\mu$ denote the traffic intensity. Then, we have:
$$
p_k = \begin{cases}
    \frac{a^k}{k!} p_0, & k \in \{0, 1, \dots, s\} \\
    \frac{a^k}{s^{k-s}(s!)} p_0, & k \in \{s+1, s+2, \dots, \} \\
\end{cases}
$$

Because the sum of the probabilities must be 1, we have:
$$
\sum_{k=0}^{\infty} p_k = p_0 \sum_{k=0}^{s} \frac{a^k}{k!} + p_0 \sum_{k=s+1}^{\infty} \frac{a^k}{s^{k-s}(s!)} = 1.
$$
Assuming that the traffic intensity is less than the number of servers, i.e., $a < s$, solving for $p_0$, we get:
$$
p_0 = \left[\sum_{k=0}^{s-1} \frac{a^k}{k!} + \frac{a^s}{s!}\left(\frac{s}{s-a}\right)\right]^{-1}.
$$

## Performance Metrics

The probability that a customer entering the system has to wait in queue is: 
$$
\begin{aligned}
\Pr(W_q > 0) & = \Pr(\{\text{number of customers in the system} \ge s\}) \\
& = \sum_{k=s}^{\infty} p_k \\ 
& = \frac{\frac{a^s}{s!}\left(\frac{s}{s - a}\right)}{\sum_{k=0}^{s-1} \frac{a^k}{k!} + \frac{a^s}{s!}\left(\frac{s}{s - a}\right)}. 
\end{aligned}
$$
This is also called the Erlang C formula. We denote it as $C(s, a)$.
