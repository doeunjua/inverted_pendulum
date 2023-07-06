# inverted_pendulum
# inverted peledulm
## 사용자 조종환경
```python

from pynput import keyboard  # pip install pynput

action = 0

def left():
    global action
    action = -2

def right():
    global action
    action = 2

def dont_accelerate():
    global action
    action = 0


listener = keyboard.GlobalHotKeys({
    'j': left,  # j는 시계 방향으로 가속
    'l': right,  # l은 반시계 방향으로 가속 
    'k': dont_accelerate  # k는 가속하지 않음
})

listener.start()


import gymnasium as gym
import time

env = gym.make('Pendulum-v1', render_mode="human")
env.reset()
steps = 0

while True:
    # env.step 진행
    _, reward, done, _, _ = env.step((action,))

    print("현재", steps, "스텝 에서의 보상 반환값:", reward)

    steps += 1
    time.sleep(0.1)
```
###  **목표: 추를 거꾸로 세우는 것이 목적**
####  특징:카트폴과 같은 이산행동공간이 아닌 연속 행동 공간이다.

## **강화학습 과정**

### **step1.필수적인 라이브러리 불러오기**
```python
import gymnasium as gym
import numpy as np
import random
from collections import deque 
from keras.layers import Dense
from tensorflow import keras
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt
```
### **step2.inverted_pendulum 생성**
```python
env = gym.make('Pendulum-v1',g=1)
```
- gym 라이브러리의 make 함수를 사용하여 'Pendulum-v1' 환경을 생성하는 코드
### **step3.output**
```python
output = 41 
```
- -2부터 2까지 41개 간격으로 액션이 나오도록 설정할 것임

### **step4. DQN class**
```python
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='tanh') #input_dim=3 : state의 차원(x,y,angular velocity)
        self.d2 = Dense(64, activation='tanh')
        self.d3 = Dense(32, activation='tanh')
        self.d4 = Dense(output, activation='linear') #continuous한 action을 출력으로 가짐
        self.optimizer = keras.optimizers.Adam(0.001)

        

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x
```

- DQN 클래스는 Keras의 Model 클래스를 상속받아서 구현.
모델의 초기화(__init__) 메서드에서는 네트워크의 구조를 정의 각 층은 Dense 레이어로 구성

```python
self.d1 = Dense(128, input_dim=3, activation='tanh')
self.d2 = Dense(64, activation='tanh')
self.d3 = Dense(32, activation='tanh')
```

- Dense 레이어는 입력 데이터를 받아서 가중치와 편향을 적용한 후, 활성화 함수를 통과시킴. 가중치와 편향은 학습 과정에서 업데이트되는 매개변수.
- units: 출력 유닛의 수.
- activation: 활성화 함수를 지정
- input_dim: 입력 차원을 지정. 첫 번째 레이어에서만 사용
- ```self.d1 = Dense(128, input_dim=3, activation='tanh')```:128개의 유닛을 가지는 dense 레이어를 생성하고 입력차원은 3차원이고 활성화 함수로는 tanh함수를 사용
- tanh 함수는 출력 범위가 -1에서 1까지로 제한. 입력의 범위에 대한 제한된 출력을 생성하므로, 출력 값이 크게 증가하거나 감소하는 것을 방지. 신경망의 안정성과 수렴 속도를 향상시키는 데 도움 됨.
```python
self.d4 = Dense(output, activation='linear')
```
- 마지막 층(self.d4)은 출력으로 output 크기의 벡터를 생성하며, 활성화 함수로는 선형 함수(linear)가 사용. 'linear' 활성화 함수를 사용하여 출력 층을 정의하는 것은 연속적인 출력과 연속적인 행동 공간을 다루는 문제에 적합하며, 값을 직접적으로 반환할 수 있는 장점을 가지고 있음.

```python
self.optimizer = keras.optimizers.Adam(0.001)
```
- Adam 옵티마이저를 생성하고 학습 과정에서 모델의 가중치를 업데이트하는 데 사용.
Adam 옵티마이저는 경사 하강법의 한 종류로, 학습 속도를 조절하는 최적화 알고리즘. 0.001은 학습 속도인 학습률(learning rate)을 나타내며, Adam 옵티마이저의 학습 속도를 조정하는 매개변수. 학습률은 가중치 업데이트의 크기를 결정하는 역할을 함.


```python
def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x
```
- call 메서드는 모델에 입력을 주고 출력을 계산하는 역할을 수행.
- 해당 코드에서는 입력 x를 각각의 Dense 레이어를 통과시켜 나온 결과를 차례대로 다음 레이어의 입력으로 전달. 
- 이는 신경망의 순전파(feedforward) 과정을 구현한 것.신경망의 순전파(Feedforward) 과정은 입력 데이터가 네트워크를 통과하여 출력을 계산하는 과정. 주어진 입력에 대해 신경망은 각 층의 뉴런을 통과하면서 가중치와 활성화 함수를 적용하여 중간 출력을 계산하고, 마지막으로 출력 층에서 최종 예측 값을 얻음. 각 레이어는 입력에 가중치를 곱하고 활성화 함수를 적용하여 출력을 계산.

- `x = self.d1(x)`: 입력 x를 첫 번째 Dense 레이어인 self.d1에 통과시킴. 이는 128개의 뉴런을 가진 레이어로서, 입력 x에 가중치를 곱하고 활성화 함수를 적용하여 결과를 계산.

- `x = self.d2(x)`: 이전 레이어의 출력인 x를 두 번째 Dense 레이어인 self.d2에 통과시킴. 이는 64개의 뉴런을 가진 레이어로서, 이전 레이어의 출력에 가중치를 곱하고 활성화 함수를 적용하여 결과를 계산.

- `x = self.d3(x)`: 이전 레이어의 출력인 x를 세 번째 Dense 레이어인 self.d3에 통과시킴. 이는 32개의 뉴런을 가진 레이어로서, 이전 레이어의 출력에 가중치를 곱하고 활성화 함수를 적용하여 결과를 계산.

- `x = self.d4(x)`: 이전 레이어의 출력인 x를 마지막 Dense 레이어인 self.d4에 통과시킴. 이는 output의 크기를 가진 레이어로서, 이전 레이어의 출력에 가중치를 곱하고 활성화 함수를 적용하여 최종 출력을 계산. 여기서는 선형 활성화 함수인 linear를 사용.

-  최종적으로 계산된 출력이 반환되어 호출자에게 전달. 이를 통해 모델은 입력에 대한 예측값을 생성.

### **step5. 파라미터 값 & 변수생성**

```python
model = DQN()#DQN모델 생성
D = deque(maxlen=4000)#크기 4000인 리플레이 버퍼
step = env.spec.max_episode_steps#step=200
studyrate = 0.9
batch_size = 64
episode = 400
eps = 0.9
eps_decay = 0.99
rewards_list = []

```
### **step6. 에피소드 반복**

```python
for i in range(episode):
    state = env.reset()[0]
    state = state.reshape(1,3)
    eps = max(0.01, eps * eps_decay)
    total_reward = 0

    for j in range(step):
        if np.random.rand() <= eps:
            action = (np.random.randint(0, output))/10+2
        else:
            action = (np.argmax(model.call(state)))/10+2 
    

        next_state, reward = env.step((action,))[0:2]
        
        next_state = next_state.reshape(1,3)
        D.append((state, action, reward, next_state))#(1x3),(1),(1),(1x3)

        state = next_state
        total_reward += reward
    rewards_list.append(total_reward)

    #DQN 학습
    if i > 20 :
        mini_batch = random.sample(D, batch_size)
        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])

        actions_to_index = np.array([int((action+2)*10) for action in actions])

        target_y = model.call(states).numpy()
        target_y[range(64),actions_to_index] = rewards + studyrate * np.max(model.call(next_states).numpy(), axis=1)

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if i % update_target_network_freq == 0:
        model.set_weights(model.get_weights())

    #10번마다 모델 저장  
    if i % 10 == 0:
        model.save_weights('pendulum_weight.h5')
    # 에피소드 당 평균 리워드 계산 및 출력
    avg_reward = total_reward / step
```
```python
state = env.reset()[0]
state = state.reshape(1,3)
eps = max(0.01, eps * eps_decay)
total_reward = 0
```
- `env.reset()`:
(array([ 0.9739883, -0.2265984,  0.169537 ], dtype=float32), {})
- `state = env.reset()[0]`: [-0.291616   -0.95653546 -0.86063063]
- `state = state.reshape(1,3)`:[[-0.291616   -0.95653546 -0.86063063]]
- `eps = max(0.01, eps * eps_decay)`:0.9에서 0.99씩 감소하는데 최소 입실론 0.01을 보장

```python
for j in range(step):
        if np.random.rand() <= eps:
            action = (np.random.randint(0, output)/10-2)
        else:
            action = (np.argmax(model.call(state))/10-2)

        next_state, reward = env.step((action,))[0:2]
        
        next_state = next_state.reshape(1,3)
        D.append((state, action, reward, next_state))#(1x3),(1),(1),(1x3)

        state = next_state #다음 state로 바꿈
        total_reward += reward 
    rewards_list.append(total_reward)
```
- 첫번째 if문에서 eps보다 random값이 작으면 무작위로 action을 선택한다. ` action = (np.random.randint(0, output)/10-2) ` 은 0부터 41까지의 정수중에서 수를 뽑고 10으로 나누고 -2를 해서 -2에서 2사이의 값이 action으로 나오도록 한다.
- else문에서 model.call(state)는 [q값,q값,q값,..........,q값]이렇게 총 41개의 q값이 나오고 argmax하면 인덱스가 나온다. 인덱스는 0부터 41까지니까 -2에서 2까지의 action이 나오게 하려면 마찬가지로 10으로 나누고 2를 빼는 과정을 해야한다.
- `next_state, reward = env.step((action,))[0:2]` 에서 다음상태와 보상을 가져온다.
- reshape를 통해서 [x,y,Angular Velocity] 를 [[x,y,Angular Velocity]]  로 바꿔줌
- 메모리 버퍼 D에 state, action, reward, next_state을 넣음

```python
 if i > 40 :
        mini_batch = random.sample(D, batch_size)
        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])

        actions_to_index = np.array([int((action+2)*10) for action in actions])
        target_y = model.call(states).numpy()
        target_y[range(batch_size), actions_to_index] = rewards + studyrate * np.max(model.call(next_states).numpy(), axis=1)

        with tf.GradientTape() as tape:
            predicted_values = model(states)
            loss = tf.reduce_mean(tf.square(target_y - predicted_values))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print('Episode: {}, Average Reward: {:.3f}, Epsilon: {:.2f}'.format(i, avg_reward, eps))
```
- 에피소드 40이후부터 학습을 시킨다.
- 메모리버퍼 D에서 batch_size만큼 가져와서 mini_batch에 넣는다
- mini_batch의 sample에서 state,action,reward,next_State를 가져온다. 
- -2에서 2까지의 action을 0에서 40까지 41개의 인덱스로 바꾸기 위해 `actions_to_index = np.array([int((action+2)*10) for action in actions])`를 해준다.
- target_y는 현재 모델이 예측한 값임. 첫 번째 줄에서 model.call(states)를 통해 현재 상태 states에 대한 예측 값을 얻음. 이후, numpy() 메서드를 사용하여 예측 값을 NumPy 배열로 변환.
- target_y[range(batch_size), actions]는 target_y 배열에서 batch_size 개의 행에서 해당하는 액션 인덱스의 위치를 선택합니다. 이 위치에 대해 새로운 값인 rewards + studyrate * np.max(model.call(next_states).numpy(), axis=1)를 할당하여 업데이트합니다. 이 값은 현재 보상(rewards)에 다음 상태(next_states)에서의 최대 예측 값을 곱한 것을 더한 것입니다. 이렇게 함으로써, target_y 배열의 선택된 위치에 올바른 타깃 값을 할당하고, 이를 기반으로 모델을 학습시키게 됩니다.
- tf.GradientTape()를 사용하여 모델의 연산을 기록하고 model(states)를 통해 현재 상태 states에 대한 모델의 예측 값을 얻음.

- 다음으로, 예측 값과 target_y 사이의 평균 제곱 오차(Mean Squared Error)를 계산하여 loss에 저장. 이는 학습을 통해 예측 값을 target_y에 가깝게 만들기 위한 오차를 나타냄.

- 그 다음, tape.gradient(loss, model.trainable_variables)를 사용하여 loss에 대한 모델의 각 변수에 대한 기울기(gradient)를 계산합니다. 이를 통해 모델의 학습 가능한 변수들에 대한 기울기를 얻을 수 있습니다.

- 마지막으로, model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))를 사용하여 기울기를 모델의 변수에 적용하여 모델을 업데이트. 이렇게 함으로써 모델은 주어진 오차에 따라 학습이 이루어짐.

```python
plt.plot(rewards_list)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
```
그래프 그림

### **결론**
![image](https://github.com/doeunjua/inverted_pendulum/assets/122878319/8c729ee1-bbfd-43c4-95e5-3583c6e0db3a)


### **전체코드**
```python
import gymnasium as gym
import numpy as np
import random
from collections import deque 
from keras.layers import Dense
from tensorflow import keras
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt

#inverted pendulum
env = gym.make('Pendulum-v1',g=1)

#action space
action_space = np.linspace(-2,2,41) # -2~2 : 41개의 action space

#인공신경망 만들기
# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(512, input_dim=3, activation='tanh') #input_dim=3 : state의 차원(x,y,angular velocity)
        self.d2 = Dense(256, activation='tanh')
        self.d3 = Dense(128, activation='tanh')
        self.d4 = Dense(len(action_space), activation='linear') #continuous한 action을 출력으로 가짐
        self.optimizer = keras.optimizers.Adam(0.001)


    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

model = DQN()

D = deque(maxlen=8000)
step = env.spec.max_episode_steps
studyrate = 0.9
batch_size = 32
episode = 500
eps = 0.9
eps_decay = 0.99
rewards_list = []

#simulation
for i in range(episode):
    state = env.reset()[0]
    state = state.reshape(1,3)
    eps = max(0.01, eps * eps_decay)
    total_reward = 0

    for j in range(step):
        if np.random.rand() <= eps:
            action = np.random.choice(action_space)
        else:
            action= action_space[np.argmax(model.call(state))]
            
        next_state, reward = env.step((action,))[0:2]
        
        
        next_state = next_state.reshape(1,3)
        D.append((state, action, reward, next_state))#(1x3),(1),(1),(1x3)

        state = next_state
        total_reward += reward
    
    
    avg_reward = total_reward / step
    rewards_list.append(avg_reward)

    #DQN 학습
    if i > 40 :
        mini_batch = random.sample(D, batch_size)
        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])

        actions_to_index = np.array([np.where(action_space == action)[0][0] for action in actions])

        target_y = model.call(states).numpy()
        target_y[range(batch_size), actions_to_index] = rewards + studyrate * np.max(model.call(next_states).numpy(), axis=1)

        with tf.GradientTape() as tape:
            predicted_values = model(states)
            loss = tf.reduce_mean(tf.square(target_y - predicted_values))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print('Episode: {}, Average Reward: {:.3f}, Epsilon: {:.2f}'.format(i, avg_reward, eps))

plt.plot(rewards_list)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
```
