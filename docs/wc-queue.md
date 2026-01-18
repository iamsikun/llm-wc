# Winner's Curse in Queueing System Capacity Planning

This document summarizes how the winner's curse, arising from the selection of a "best" Large Language Model (LLM), propagates to capacity planning in a queueing system, leading to systematic investment distortions.

## Capacity Planning in Queueing Systems

We consider a customer service center where arrivals follow a Poisson process with rate $\lambda$. Incoming requests are first handled by an LLM agent. If the LLM fails, the request is routed to a human agent. This constitutes a sequential filtration system. If the LLM has a success probability $p$, the effective arrival process to the human agents is a thinned Poisson process with rate $\lambda_{\text{eff}} = \lambda(1-p)$.

When a request reaches a human agent, the service time is a random variable. The firm's problem is to choose a capacity level to minimize costs, subject to a Service Level Agreement (SLA). The SLA can constrain various performance metrics, such as the expected time a customer wait before receiving service or the probability of waiting beyond a certain duration.

Let's assume the service time follows an exponential distribution with rate $\mu$. The firm's decision variable is $\mu$. The cost of capacity is $w\mu$. Consider an SLA that the expected time a customer wait before receiving service, $p\mathbb{E}[W_q]$, must not exceed $\tau$. The optimization problem is:
$$
\begin{aligned}
\min_{\mu \geq 0}\; & w\mu \\ 
\text{s.t.}\; & p\mathbb{E}[W_q] \le \tau. 
\end{aligned}
$$
The expected waiting time in queue is $p\mathbb{E}[W_q]$ because the probability of the request being routed to a human agent is $p$. If the customer's request is resolved by the LLM, the waiting time is 0.
With Poisson arrivals and exponential service times, this is a standard M/M/1 queue. Tshe expected waiting time in the queue is $\mathbb{E}[W_q] = \lambda_{\text{eff}}[\mu(\mu - \lambda_{\text{eff}})]^{-1}$. Since $\mathbb{E}[W_q]$ monotonically decreases in $\mu$, the cost-minimizing firm will choose $\mu$ to bind the constraint:
$$
\mu^* = \frac{\lambda_{\text{eff}}}{2} + \sqrt{\left(\frac{\lambda_{\text{eff}}}{2}\right)^2 + \frac{p\lambda_{\text{eff}}}{\tau}}.
$$
To determine the optimal capacity $\mu^*$, the firm needs to know the LLM's success probability $p$.

## The Winner's Curse in LLM Selection

In practice, a firm evaluates a set of $K$ LLM configurations. For each configuration $k \in \{1, \dots, K\}$, the true but unknown success probability is $p_k$. The firm estimates these probabilities by running $N$ pilot tests, obtaining estimates $\hat{p}_k$. The firm then selects the configuration with the highest estimated performance, $\hat{k}^* \in \argmax_{k} \hat{p}_k$, and uses its estimated success probability, $\hat{p}_{\hat{k}^*}$, for capacity planning.

However, due to random sampling error, the estimated performance of the winner is likely an overestimation of its true performance. This phenomenon is the winner's curse. Formally, the expected bias is non-negative:
$$
\text{WC} = \mathbb{E}[\hat{p}_{\hat{k}^*} - p_{\hat{k}^*}] \ge 0,
$$
where the expectation is over the randomness in the pilot tests. This over-optimism leads the firm to underestimate the true arrival rate of customers to the human agents. The firm plans for an arrival rate of $\hat{\lambda}_{\text{eff}} = \lambda(1 - \hat{p}_{\hat{k}^*})$, but the actual rate is $\lambda_{\text{eff}} = \lambda(1 - p_{\hat{k}^*})$. On average, $\mathbb{E}[\hat{\lambda}_{\text{eff}}] < \mathbb{E}[\lambda_{\text{eff}}]$. This misjudgment distorts the capacity decision, leading to under-provisioning.

## Capacity Distortion

We now analyze the performance degradation resulting from this under-provisioning. The distortion is the difference between the actual performance metric and the SLA target.

### $M/M/1$ queue with expected waiting time constraint

The SLA requries that the expected waiting time in queue should not exceed $\tau$. For an M/M/1 queue, $\mathbb{E}[W_q] = \frac{\lambda_{\text{eff}}}{\mu(\mu-\lambda_{\text{eff}})}$. The optimal service rate $\mu^*$ that binds the constraint is found by solving a quadratic equation:
$$
\mu^*(p) = \frac{\lambda_{\text{eff}}}{2} + \sqrt{\left(\frac{\lambda_{\text{eff}}}{2}\right)^2 + \frac{p\lambda_{\text{eff}}}{\tau}}.
$$
The firm sets its capacity to $\hat{\mu}^* = \mu^*(\hat{p}_{\hat{k}^*})$.

The actual expected waiting time is $p_{\hat{k}^*}\mathbb{E}[W_{q, \text{actual}}\mid\mathcal{D}] = p_{\hat{k}^*}\frac{\lambda_{\text{eff}}}{\hat{\mu}^*(\hat{\mu}^* - \lambda_{\text{eff}})}$.
The expectation is taken over the arrival and service times, conditioned on the data $\mathcal{D}$ in the pilot tests. 
We can express the actual performance in terms of the target $\tau$ and the estimation error. The firm's choice $\hat{\mu}^*$ satisfies $\hat{\mu}^*(\hat{\mu}^*-\hat{\lambda}_{\text{eff}}) = \hat{p}_{\hat{k}^*}\hat{\lambda}_{\text{eff}}/\tau$. Substituting this into the expression for $W_{q, \text{actual}}$:
$$
p_{\hat{k}^*}\mathbb{E}[W_{q, \text{actual}}\mid\mathcal{D}] = \frac{p_{\hat{k}^*}\lambda_{\text{eff}}}{\hat{\mu}^*(\hat{\mu}^* - \hat{\lambda}_{\text{eff}} - (\lambda_{\text{eff}} - \hat{\lambda}_{\text{eff}}))} = \frac{p_{\hat{k}^*}\lambda_{\text{eff}}}{\hat{p}_{\hat{k}^*}\hat{\lambda}_{\text{eff}}/\tau - \hat{\mu}^*(\lambda_{\text{eff}} - \hat{\lambda}_{\text{eff}})}.
$$
Let $wc = \hat{p}_{\hat{k}^*} - p_{\hat{k}^*}$ be the estimation error for the chosen model. Then $\lambda_{\text{eff}} - \hat{\lambda}_{\text{eff}} = \lambda wc$. This gives:
$$
p_{\hat{k}^*}\mathbb{E}[W_{q, \text{actual}}\mid\mathcal{D}] = \frac{\tau p_{\hat{k}^*}\lambda_{\text{eff}}}{\hat{p}_{\hat{k}^*}\hat{\lambda}_{\text{eff}} - \tau \hat{\mu}^* \lambda wc}
$$

### $M/M/c$ queue with expected waiting time constraint

The logic is identical to the previous case, but with the constraint being on the expected waiting time in queue, $p\mathbb{E}[W_q] \le \tau$. The expected waiting time is
$$
p\mathbb{E}[W_q] = \frac{pC(c, \lambda_{\text{eff}}/\mu)}{c\mu - \lambda_{\text{eff}}}.
$$
Here, $C(s, a)$ is the Erlang C formula, which gives the probability that an arriving customer must wait for service in an M/M/c queue with $s$ servers. It is defined as:
$$
C(s, a) = \frac{\frac{a^s}{s!}\left(\frac{s}{s - a}\right)}{\sum_{k=0}^{s-1} \frac{a^k}{k!} + \frac{a^s}{s!}\left(\frac{s}{s - a}\right)}. 
$$

Using diffusion approximation (Halfin and Whitt, 1981), the number of servers can be parameterized as: 
$$
c \approx a + \beta \sqrt{a},
$$
where $a = \lambda_{\text{eff}}/\mu$ is the traffic intensity. The parameter $\beta$ is the safety factor, determining how much slack is added to the capacity. The decision variable thus switches from $c$ to $\beta$. The probability of waiting, i.e., the Erlang C formula, can be approximated as: 
$$
\Pr(W_q > 0) \approx \alpha_{HW}(\beta) = \alpha_{HW}(\beta) = \left[ 1 + \frac{\beta \Phi(\beta)}{\phi(\beta)} \right]^{-1}. 
$$
$\Phi(\beta)$ is the cumulative distribution function of the standard normal distribution. The expected waiting time is then:
$$
p\mathbb{E}[W_q] \approx \frac{p\alpha_{HW}(\beta)}{\beta \mu \sqrt{a}}.
$$

Thus, the optimal $\hat{\beta}^*$ is found by solving the equation:
$$
\frac{\hat{p}_{\hat{k}^*}\alpha_{HW}(\hat{\beta}^*)}{\hat{\beta}^*} = \tau  \sqrt{\hat{\lambda}_{\text{eff}}\mu}.
$$
The optimal capacity is then: 
$$
\hat{c}^* = \hat{\lambda}_{\text{eff}} / \mu + \hat{\beta}^* \sqrt{\hat{\lambda}_{\text{eff}}/\mu}.
$$