
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the First-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and First-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) First-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / First-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Q learning is TD control learning where Q-value is updated by a bootstrapped estimation and adjusts " \
             "incrementally, the formulation of the updating for Q-value is ‘Q(S_t, A_t) = Q(S_t, A_t) + α(R_(t+1) + " \
             "gamma *  max_a Q(S_(t+1), a) - Q(S_t, A_t))’. However, gamma will effects First-visit MC more because " \
             "MC estimates the expected return G_t = R_(t+1) + gamma *  R_(t+2) + ... + gamma *  ^(T-t-1)R_T, " \
             "where T is the final time step. MC relies on full trajectories and is more sensitive to gamma's " \
             "influence on future rewards."
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In Reinforce, which learning rate achieves the highest mean returns at the end of training?
    a) 6e-1
    b) 6e-2
    c) 6e-3
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.75
    b) 0.25
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "b"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.75
    c) 0.001
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.990?
    a) 0.990
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "a"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The epsilon linear decay strategy directly scales the epsilon value with the number of time steps by " \
             "setting the exploration rate. It is easier to adjust the exploration duration for training agents in " \
             "various environments. However, a decay strategy based on a decay rate parameter is sensitive to the " \
             "choice of the decay rate because the decay rate determines how quickly the epsilon decreases. However, " \
             "the optimal choice of this rate is environment-dependent and may not generalize well across different " \
             "environments because this approach is less intuitive to control the duration that epsilon will " \
             "decrease. "
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "In traditional supervised learning, our goal is to minimize the loss between the prediction and target, " \
             "and the target is usually fixed in the training dataset. As the training steps grow, we expect the loss " \
             "to decrease through training. However, in DQN, although our goal is also to minimize the predicted Q " \
             "value and the target Q value, the target value is not constant during training, and then the loss might " \
             "increase temporarily even though the agent's performance improves because the critic network needs to " \
             "adjust its parameters to match the new target value. However, over time, as the network continues to " \
             "learn, the loss function should decrease, and the agent's performance should improve. This is because " \
             "the network is gradually learning to approximate the optimal Q-value function, and the target Q-value " \
             "becomes a better estimate of the true Q-value function."
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The occurrence of a spike at approximately 2000 steps with one spike is linked to the update frequency " \
             "of the target network. Whenever the target network is updated, there is a sudden change in its " \
             "estimated target Q-values, which results in a larger discrepancy with the critic network's estimates. " \
             "This inconsistency leads to higher loss values since the critic network must adapt to the new targets. " \
             "As the critic network converges to the target, the loss decreases until the next update produces a " \
             "spike. This process recurs throughout the training phase, leading to regular spikes in loss curves."
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "In Q4, we trained the model using the optimal architecture of [512, 256] for both the actor and critic " \
             "networks, as determined by the sweep result. For Q5, we use the same architecture instead of " \
             "experimenting with various parameter sets. During the fine-tuning stage, we began with a grid search " \
             "using 5 seeds to identify the proper direction for tuning. We set gamma to range between 0.97 and " \
             "0.990, tau between 0.001 and 0.2, and the learning rate between 1e-4 and 1e-3. With three search " \
             "samples for each parameter and the same learning rate for both networks, we ran the sweep training. The " \
             "grid search revealed that the best parameters are gamma=0.98, tau=0.105, and learning rate=5.5e-4. " \
             "Next, we performed a random search with a uniform distribution, yielding tau=0.1, gamma=0.98, " \
             "and lr of 5e-4 for the actor network. However, the performance was inconsistent across different runs " \
             "using the same settings, indicating that it was not yet optimal. To address this, we implemented " \
             "hyperparameter scheduling for the standard deviation (std) of the action and increased the batch size. " \
             "After tuning the scheduling parameters for two more runs, we achieved an optimal agent with an average " \
             "score of about 300."
    return answer
