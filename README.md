# NAPIOD
## Non-average price impact in order-driven markets

`napiod` gives a python implementation of the measurement of price impact in order-driven market introduced in [Bellani et al. (2021)](https://arxiv.org/abs/2110.00771).
This measurement of price impact quantifies the extent to which price movements are caused by a labelled agent in the market, either by her own market orders (direct impact) or by other market participants that react to her actions (indirect impact). 
The price impact is scenario-specific and depends both on the agent's behaviour and on the market environment.
The measurement of price impact is based on a model that utilises state-dependet Hawkes processes introduced in
[Morariu-Patrichi and Pakkanen (2017)](https://arxiv.org/abs/1707.06970) and [Morariu-Patrichi and Pakkanen (2018)](https://arxiv.org/abs/1809.08060), and so `napiod` depends on [mpoints](https://github.com/maximemorariu/mpoints).



## The model 

We consider four streams of random times: 
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

The four sequences of random times give rise to a four-dimensional counting process 
$N=(N_1, N_2, N_3, N_4)$ 
with the following interpretation of its components:
* $N_1(t)$  denotes the number of seller-initiated trades that happened before or at time $t$
(identified with the number of market orders arrived on the bid side of the order book by time $t$);
* $N_2(t)$
 denotes the number of buyer-initiated trades that happened before or at time $t$
(identified with the number of market orders arrived on the ask side of the order book by time $t$);
* $N_3(t)$ denotes the number of decreases in the mid-price 
	 caused by a limit order insertion or cancellation that happened before or at time $t$;
* $N_4(t)$ denotes the number of increases in the mid-price 
	 caused by a limit order insertion or cancellation that happened before or at time $t$;




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
If you are contributing to `mpionts`, you might want to install it in development mode.
See [development-mode](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#development-mode)

```
pip install --upgrade --force-reinstall --editable <path-to-mpoints-root>
```
