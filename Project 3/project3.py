#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# from socketserver import BaseRequestHandler
import numpy as np
import random 
import math
import pickle

random.seed(10)

#2-12 -> 2-10 for each numbered card, 11 to represent an ace
full_deck =  list(range(2,12))*4 + [10,10,10]*4
round_deck = list(full_deck)
#print(full_deck)
player_action = [1,0] #1 for hit, 0 for stand - note if player going second stands, then the game has ended and we can compare

#Call this to replenish the deck at the start of every round
def replaceDeck():
    global round_deck 
    round_deck= list(full_deck)

#Takes a card, without replacement, from the round_deck
#Returns the value drawn of that card]
def drawCard():
    drawedCard = np.random.choice(round_deck)
    round_deck.remove(drawedCard)
    return drawedCard



class State:
    def __init__(self):
    ##define player state values
    #sum_player_value = sum of player card values
    #dealer_shown_card = the card value of the dealer's shown card
    #num_of_aces = number of aces the player has in his/her hand
        self.data = np.zeros((3,), dtype=int) #this is originally empty, is composed of the elements as follows:
            # 1) int-The sum of the values of cards that the player (RL agent) has
            # 2) int-The value of the shown_dealer card that the dealer has face up
            # 3) int-The number of aces a player current has. 0 indicates the player has no ace.

        #self.hash_val = None
        self.end = False
        self.dealer_sum = 0
        self.num_of_dealer_aces = 0
        

    def next_state(self, action):
        # get current state items
        current_sum = self.data[0]
        shown_dealer_card = self.data[1]
        num_aces = self.data[2]

        #actually perform the action passed
        #hit
        if action:
            card = drawCard()
            if card == 11:
                num_aces+= 1
            
            current_sum += card
        else: #stand
            self.end = True
            self.data = [current_sum, shown_dealer_card,num_aces]

        if current_sum >= 21:
            if current_sum > 21:
                if num_aces > 0:
                    current_sum -= 10
                    num_aces-=1
                else:
                    self.end = True
                    self.data = [current_sum, shown_dealer_card,num_aces]
            else:
                self.end = True
                self.data = [current_sum, shown_dealer_card,num_aces]
        #otherwise game isnt necessarily ended
        self.data = [current_sum, shown_dealer_card,num_aces]

        # since RL opponent goes first, we can print their sum of cards
        # we also print our sum of cards
    def print_state(self):
        print("AI Player sum: ", self.data[0])
        #print("Dealer shown card: ", self.data[1])
        #print("Player aces: ", self.data[2])
        print("Your sum: ", self.dealer_sum)
        print("Your aces: "+ str(self.num_of_dealer_aces) + "\n")


# Class that specifies when does the game starts, who moves first, and who wins
# In your own project, this class should be changed based on the game of your choice
class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None

    def winner(self, player_value, dealer_value):
        if player_value > 21:
            if dealer_value > 21:
                #draw
                return 0
            else:
                #player busts so loss
                return -1
        #meaning player hasn't busted here
        else:
            if dealer_value>21:
                #dealer busts, so win
                return 1
            else:
                if player_value < dealer_value:
                    #player loses here
                    return -1
                elif player_value > dealer_value:
                    return 1
                else:
                    #draw
                    return 0

    
    def reset(self):
        self.p1.reset()
        self.p2.reset()
        #self.p1, self.p2 = (self.p2, self.p1)

    # def alternate(self):
    #     while True:
    #         yield self.p1
    #         yield self.p2

    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        #alternator = self.alternate()

        current_state = State()
        replaceDeck()
        # draw 2 for player 1
        # draw 1 for player 2 #dealer
        # current_state.data[0] = sum of 2 cards for p1
        # current_state.data[1] = dealer shown card
        # current_state.data[2] = num of aces that player has 
        player_sum = 0
        num_of_player_aces = 0
        num_of_dealer_aces = 0
        #Draw 2 cards
        for x in range(2):
            card_drawn = drawCard()
            if card_drawn == 11:
                num_of_player_aces+=1
            player_sum += card_drawn
        #this means that two aces were drawn:
        if player_sum == 22:
          player_sum-=10
          num_of_player_aces-=1
        current_state.data[0] = player_sum
        current_state.data[2] = num_of_player_aces

        shown_dealer_card = drawCard()
        #print("SHOWN DEALER CARD ", shown_dealer_card)
        current_state.data[1] = shown_dealer_card
        #current_state.dealer_sum = shown_dealer_card
        #print(shown_dealer_card)
        #print(num_of_player_aces)
        if shown_dealer_card == 11:
            num_of_dealer_aces = 1
        current_state.num_of_dealer_aces = num_of_dealer_aces

        #draw another card for dealer but do not show the player
        secondCard = drawCard()
        if secondCard == 11:
            num_of_dealer_aces += 1

        dealer_sum = shown_dealer_card + secondCard
        #this means that two aces were drawn:
        if dealer_sum == 22:
            dealer_sum-=10
            num_of_dealer_aces-=1
        if(isinstance(self.p2, HumanPlayer)):
            #end the game (without learning) if either player draws a natural (21)
            if dealer_sum == 21 and not player_sum == 21:
              current_state.print_state()
              return -1
            if player_sum == 21 and not dealer_sum == 21:
              current_state.print_state()
              return 1
            if player_sum == 21 and dealer_sum == 21:
              current_state.print_state()
              return 0
        else:  
          #end the game (without learning) if either player draws a natural (21)
          if dealer_sum == 21 and not player_sum == 21:
              return -1
          if player_sum == 21 and not dealer_sum == 21:
              return 1
          if player_sum == 21 and dealer_sum == 21:
              return 0

        current_state.dealer_sum = dealer_sum
        current_state.num_of_dealer_aces = num_of_dealer_aces
        self.p1.set_state(current_state)

        if print_state:
            current_state.print_state()

        #enter player logic
        while True:
            #self.p1.state.print_state()
            action = self.p1.act()
            #ONLY record values if the decision is non-trivial (sum is above 12 so bustable cards)
            if current_state.data[0] < 22:
              if current_state.data[0] > 11:
                  recordState = [self.p1.state.data, action]
                  self.p1.player_action_list.append(recordState)

            current_state.next_state(action)
            #print(current_state.data)
            self.p1.set_state(current_state)
            if current_state.end:
                break
        
        if(isinstance(self.p2, HumanPlayer)):
          current_state.print_state()

        self.p2.set_state(current_state)

        if(current_state.data[0] <= 21):
          #dealer logic - not human player yet       
          if(isinstance(self.p2, Dealer)):
              dealer_end = 0
              #run dealer until we find its state end as 1
              while dealer_end == 0:
                  dealer_end = self.p2.dealer()
          elif(isinstance(self.p2, HumanPlayer)):
              human_end = 0
              #run human player
              while human_end == 0:
                  human_end = self.p2.Human()
        else:
           if(isinstance(self.p2, HumanPlayer)):
              print("Player busted")
            
        #print(current_state.data[0])
        #call judger.winner to decide who won
        #TODO insert the actual sum of dealer cards in DEALER_VALUE
        winner_value = self.winner(current_state.data[0], self.p2.state.dealer_sum)
        self.p1.giveCredit(winner_value)
        self.reset()
        return winner_value


# AI player
# The strategies of players are implemented here. The following is the implementation of the \eps-greedy RL algorithm
# In your own project, this class should be changed based on the game of your choice and the algorithm you use
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = {}
        #initialize estimations values
        for i in range(12,22):
            for j in range(2,12):
                for k in range(0,5):
                    self.estimations[(i,j,k)] = {}
                    for a in [1,0]:
                        if i==21 and a==0:
                            self.estimations[(i,j,k)][a] = 1
                        else:
                            self.estimations[(i,j,k)][a] = 0

        self.step_size = step_size
        self.epsilon = epsilon
        self.state = None;        
        self.greedy = []
        self.symbol = 0
        self.player_action_list = []

    def reset(self):
        self.state = None
        self.greedy = []
        self.player_action_list = []

    def set_state(self, state):
        self.state = state
        #self.state.print_state
        self.greedy.append(True)

    #the winner_Value is taken from calling self.winner(state.data[0], dealer_sum) in the judger class
    def giveCredit(self, winner_value):
        for i in reversed(self.player_action_list):
            state_data, action = i[0], i[1]
            reward = self.estimations[tuple(state_data)][action] + self.step_size*(winner_value - self.estimations[tuple(state_data)][action])
            self.estimations[tuple(state_data)][action] = round(reward, 3)


    # choose an action based on the state
    # This is where epsilon-greedy is implemented 
    def act(self):
        player_sum = self.state.data[0]

        #if player_sum is <=11, we should hit
        if player_sum <=11:
            return 1
        
        if player_sum > 21:
          return 0

        # otherwise, we have to use epsilon
        # With probability epsilon, we select a random action 
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice([1,0])
            #action.append(self.symbol)
        else:
            #greedy action here, which is to hit...
            comparison = -100
            action = 0 #defaults to do nothing
            #self.state.print_state()
            for a in self.estimations[(self.state.data[0],self.state.data[1],self.state.data[2])]:
                if self.estimations[(self.state.data[0],self.state.data[1],self.state.data[2])][a]>comparison:
                    action = a
                    comparison = self.estimations[(self.state.data[0],self.state.data[1],self.state.data[2])][a]
        return action

    def save_policy(self):
        with open('policy_%s.bin', 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin', 'rb') as f:
            self.estimations = pickle.load(f)


# make a dealer class for blackjack that hits until they have 17 or more or busts
class Dealer:
    def __init__(self):
        self.state = None
        self.dealer_end = 0

    def reset(self):
        self.state = None
        self.dealer_end = 0

    def set_state(self, state):
        self.state = state

    #returns 1 if dealer wants to hit, 0 if dealer stands
    def act(self):
        if self.state.dealer_sum < 17:
            return 1
        else:
            return 0
    
    def dealer(self):
        action = self.act()
        #same logic as NextState for a Player except our action is predetermined (hit if below 17 and stand otherwise)
        if action:
            card = drawCard()
            if card == 11:
                self.state.num_of_dealer_aces += 1
            self.state.dealer_sum += card
        else: #stand
            self.dealer_end = True

        if self.state.dealer_sum > 21:
            if self.state.num_of_dealer_aces > 0:
                self.state.dealer_sum -= 10
                self.state.num_of_dealer_aces -=1
            else:
                self.dealer_end = True
        
        return self.dealer_end


# human interface
# This class allows us humans to play with AI agent
# In your project, this class should be modified based on the game of your choice
class HumanPlayer:
    def __init__(self, **kwargs):
          self.symbol = None
          self.keys = ['s', 'h']
          self.state = None
          self.human_end = False

    def reset(self):
          pass

    def set_state(self, state):
          self.state = state

    def act(self):
          key = input("Press 'h' to hit or press 's' to stand:")
          while True:
              try:
                  data = self.keys.index(key)
                  break
              except KeyError:
                  print("Please enter 'h' or 's' for hit or stand")
                  key = input()
          return data
          
    def Human(self):
          action = self.act()
          #same logic as NextState for a Player except our action is predetermined (hit if below 17 and stand otherwise)
          if action:
              card = drawCard()
              if(card == 11):
                print("Drew an Ace")
              elif(card == 10):
                print("Drew a", np.random.choice(['10', 'Jack', 'Queen', 'King'], p=[0.25, 0.25, 0.25, 0.25]))
              else:
                print("Drew a", card)
              
              if card == 11:
                  self.state.num_of_dealer_aces += 1
              self.state.dealer_sum += card
          else: #stand
              self.human_end = True
                
          if self.state.dealer_sum > 21:
              if self.state.num_of_dealer_aces > 0:
                  self.state.dealer_sum -= 10
                  self.state.num_of_dealer_aces -=1
              else:
                  self.state.print_state()
                  print("You busted :(")
                  self.human_end = True
                  return self.human_end

          self.state.print_state()
          
          return self.human_end

# Training phase
# epochs is the number of games to play during the training phase
def train(epochs, print_every_n=500):
  player1 = Player(epsilon=0.01)
  player2 = Dealer()
  judger = Judger(player1, player2)
  player1_win = 0.0
  player2_win = 0.0
  for i in range(1, epochs + 1):
    winner = judger.play(print_state=False)
    if winner == 1:
      player1_win += 1
    if winner == -1:
      player2_win += 1
    if i % print_every_n == 0:
      print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
      #player1.backup()
      #player2.backup()
      judger.reset()
    player1.save_policy()

# This function allows two AI to complete against each other (after the training phase)
def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Dealer()
    judger = Judger(player1, player2)
    player1.load_policy()
    #player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie, which is why you will see the winning rate goes to zero (i.e., game ends with a tie) when running the code.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        print("Starting new round")
        player1 = Player(epsilon=0)
        player2 = HumanPlayer()
        judger = Judger(player1, player2)
        player1.load_policy()
        winner = judger.play()
        if winner == 1:
            #means RL player wins
            print("You lose!\n")
        elif winner == -1:
            #dealer wins
            print("You win!\n")
        else:
            print("It is a tie!\n")

"""Main: we call train() in HumanPlayer and define our epochs as 10,000 iterations, likewise 10,0000 for compete().

**To stop main and see results, manually interrupt the execution the following code cell and start the next code cells**
"""

if __name__ == '__main__':
    #print("Start Train Phase")
    train(int(1e4)) # training phase first
    #print("Start Compete Phase")
    compete(int(1e4)) # then two AI complete against each other
    print("Start Play")
    play() # Finally, you can try to play against an AI :-)


from matplotlib import pyplot as plt
# AI player
# The strategies of players are implemented here. The following is the implementation of the \eps-greedy RL algorithm
# In your own project, this class should be changed based on the game of your choice and the algorithm you use
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size, epsilon):
        self.estimations = {}
        #initialize estimations values
        for i in range(12,22):
            for j in range(2,12):
                for k in range(0,5):
                    self.estimations[(i,j,k)] = {}
                    for a in [1,0]:
                        if i==21 and a==0:
                            self.estimations[(i,j,k)][a] = 1
                        else:
                            self.estimations[(i,j,k)][a] = 0

        self.step_size = step_size
        self.epsilon = epsilon
        self.state = None;        
        self.greedy = []
        self.symbol = 0
        self.player_action_list = []

    def reset(self):
        self.state = None
        self.greedy = []
        self.player_action_list = []

    def set_state(self, state):
        self.state = state
        #self.state.print_state
        self.greedy.append(True)

    #the winner_Value is taken from calling self.winner(state.data[0], dealer_sum) in the judger class
    def giveCredit(self, winner_value):
        for i in reversed(self.player_action_list):
            state_data, action = i[0], i[1]
            reward = self.estimations[tuple(state_data)][action] + self.step_size*(winner_value - self.estimations[tuple(state_data)][action])
            self.estimations[tuple(state_data)][action] = round(reward, 3)


    # choose an action based on the state
    # This is where epsilon-greedy is implemented 
    def act(self):
        player_sum = self.state.data[0]

        #if player_sum is <=11, we should hit
        if player_sum <=11:
            return 1
        if player_sum > 21:
            return 0
        # otherwise, we have to use epsilon
        # With probability epsilon, we select a random action 
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice([1,0])
            #action.append(self.symbol)
        else:
            #greedy action here, which is to hit...
            comparison = -999
            action = 0 #defaults to do nothing
            #self.state.print_state()
            for a in self.estimations[(self.state.data[0],self.state.data[1],self.state.data[2])]:
                if self.estimations[(self.state.data[0],self.state.data[1],self.state.data[2])][a]>comparison:
                    action = a
                    comparison = self.estimations[(self.state.data[0],self.state.data[1],self.state.data[2])][a]
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)
    def new_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump([], f)

#define a new train() for plotting purposes

def train(epochs, epsilon_rate, step_size_rate, print_every_n=500):
  player1 = Player(step_size = step_size_rate, epsilon=epsilon_rate)
  player1.new_policy()
  epoch_list = []
  winrate_player1 = []
  winrate_player2 = []
  player2 = Dealer()
  judger = Judger(player1, player2)
  player1_win = 0.0
  player2_win = 0.0
  for i in range(1, epochs + 1):
    winner = judger.play(print_state=False)
    if winner == 1:
      player1_win += 1
    if winner == -1:
      player2_win += 1
    if i % print_every_n == 0:
      #print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
      epoch_list.append(i)
      winrate_player1.append(player1_win / i)
      winrate_player2.append(player2_win / i)
      #player1.backup()
      #player2.backup()
      judger.reset()
    player1.save_policy()
  plt.figure()
  plt.plot(epoch_list, winrate_player1, label='RL Agent winrate')
  plt.plot(epoch_list, winrate_player2, label='Dealer winrate')
  plt.legend()
  plt.xlabel("Epochs")
  plt.ylabel("Win Rate")
  plt.title('Step Size: '+str(step_size_rate)+'\nEpsilon Rate: '+str(epsilon_rate))
  plt.show()

# This function allows two AI to complete against each other (after the training phase)
def compete(turns):
    player1 = Player(epsilon=0, step_size=0)
    player2 = Dealer()
    judger = Judger(player1, player2)
    player1.load_policy()
    #player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))

step_size_list = [0.1,0.3,0.5, 0.7, 0.9]
epsilon_list = [0.1,0.3,0.5, 0.7, 0.9]


for i in step_size_list:
  for j in epsilon_list:
    train(int(1e4), i, j)
    print("Compete")
    print("step_size: "+str(i) +"\nEpsilon: "+str(j))
    compete(int(1e4))

