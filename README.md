# NAPIOD
## Non-average price impact in order-driven markets

`napiod` gives a python implementation of the measurement of price impact in order-driven markets introduced in [Bellani et al. (2021)](https://arxiv.org/abs/2110.00771).
This measurement of price impact quantifies the extent to which price movements are caused by a labelled agent in the market, either by her own market orders (direct impact) or by other market participants reacting to her actions (indirect impact). 
The price impact is scenario-specific and depends both on the agent's behaviour and on the market environment.
The measurement of price impact is based on a model that utilises state-dependet Hawkes processes introduced in
[Morariu-Patrichi and Pakkanen (2017)](https://arxiv.org/abs/1707.06970) and [Morariu-Patrichi and Pakkanen (2018)](https://arxiv.org/abs/1809.08060), and so `napiod` depends on [mpoints](https://github.com/maximemorariu/mpoints).



## The model 

We consider five streams of random times: 
the stream 
$T^{0}_1, T^{0}_2, \dots$ 
of times when our labelled agent submits market orders;
the stream 
$T^{1}_1, T^{1}_2, \dots$ 
of times when limit orders are executed on the bid side 
(equivalently identified with the arrival times of sell market orders);
the stream  
$T^{2}_1, T^{2}_2, \dots$ 
of times when limit orders are executed on the ask side 
(equivalently identified with arrival times of buy market orders); 
the stream 
$T^{3}_1, T^{3}_2, \dots$ 
of times when 
either an ask limit order is inserted inside the spread,
or the cancellation of a bid limit order depletes the liquidity available at the first bid level;
the stream 
$T^{4}_1, T^{4}_2, \dots$ 
of times when 
either a bid limit order is inserted inside the spread,
or the cancellation of an ask limit order depletes the liquidity available at the first ask level.

The five sequences of random times give rise to a five-dimensional counting process 
$N=(N_0, N_1, N_2, N_3, N_4)$ 
with the following interpretation of its components:
* $N_0(t)$ denotes the number of market orders that our labelled agent has submitted before or at time $t$;
* $N_1(t)$  denotes the number of seller-initiated trades that happened before or at time $t$
(identified with the number of market orders arrived on the bid side of the order book by time $t$);
* $N_2(t)$
 denotes the number of buyer-initiated trades that happened before or at time $t$
(identified with the number of market orders arrived on the ask side of the order book by time $t$);
* $N_3(t)$ denotes the number of decreases in the mid-price 
	 caused by a limit order insertion or cancellation that happened before or at time $t$;
* $N_4(t)$ denotes the number of increases in the mid-price 
	 caused by a limit order insertion or cancellation that happened before or at time $t$


The counting process $N$ is paired with the state variable $X = (X_1, X_2)$. 
At time $t$, the state variable $X(t)$ summarises the configuration 
of the limit order book at time $t$, 
by recording a proxy for the volume imbalance, 
and the variation of the mid-price compared to time $t-$. 
More precisely, 
$X_1(t) = -1$ if the volumes imbalance at time $t$ is -33% or more negative; 
$X_1(t) = 0$ if the volumes imbalance at time $t$ is between -33%  and +33%; 
$X_1(t) = +1$ if the volumes imbalance at time $t$ is +33% or more positive; 
$X_2(t) = -1$ if the latest event in the order book has decreased the mid-price;
$X_2(t) = 0$ if the latest event in the order book has left the mid-price unchanged;
$X_2(t) = +1$ if the latest event in the order book has increased the mid-price.

The pair $(N, X)$ is modelled as a state-dependent Hawkes process. 

Each agent's market order $T^{0}_j$ has two effects.
On the one hand, at every $T^{0}_j$ the state variable $X$ is updated, and the mid-price decreases if $X(T^{0}_j) = -1$ or it increases if $X(T^{0}_j) = +1$.
On the other hand, every $T^{0}_j$ alters the intensity of occurrence of the random times $T^{1}$, $T^2$, $T^3$, $T^4$, 
and these in turn will produce updates of the state process $X$. 
A quantification of the first effect produces our measurement of direct price impact, 
and a quantification of the second effect produces our mesurement od indirect price impact. 
Direct price impact is non-negative and non-decreasing, and the larger it is the more unfavourably the mid-price moves against the agent.
Indirect impact, instead, can be positive or negative:
it is positive when market participants react to the agent and move the mid-price in a way unfavourable to the agent;
it is negative when market participants react to the agent and move the mid-price in a way favourable to the agent. 

For more detail into the model and the measurement of price impact, please consult [Bellani et al. (2021)](https://arxiv.org/abs/2110.00771).


## Installation

```
pip install napiod
```

### Build from source
#### Build the wheel using `setuptools`. 
```
pip install -q build
python -m build
```

#### Install in development mode
If you are contributing to `napiod`, you might want to install it in development mode.
See [development-mode](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#development-mode)

```
pip install --upgrade --force-reinstall --editable <path-to-napiod-root>
```
