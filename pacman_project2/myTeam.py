# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import pickle
import game
from os import path

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent', **args):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  numTraining = 0
  if 'numTraining' in args:
    numTraining = args['numTraining']
  return [eval(first)(firstIndex, numTraining), eval(second)(secondIndex, numTraining)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  #epsilon=0.05, alpha=0.2, gamma=0.8
  def __init__(self, index, numTraining = 0):
    CaptureAgent.__init__(self, index)
    self.numTraining = numTraining
    self.currentEpisode = 0
    self.epsilon = 0.05
    self.alpha = 0.2
    self.gamma = 0.8
    self.lastAction = None
    self.episodeRewards = 0.0
    if self.numTraining == 0 and path.exists('weights'):
        with open('weights' + str(self.index), 'rb') as fp:
          self.weights = pickle.load(fp)
          print ("Loaded weights")
    else:
        self.weights = {
          'successorScore': 100.0,
          'distanceToFood': -1.0,
          'invaderDistance': -2.0,
          'ghostAlert': -5.0,
          #'distanceToCapsule': -2.0,
          'invaderAlert': -2.0
        }

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):

    self.observationHistory.append(gameState)

    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    '''
    You should change this in your own agent.
    '''

    if util.flipCoin(self.epsilon) and self.isTraining():
      action = random.choice(actions)
    else:
      action = self.getBestActionByQValue(gameState, actions)

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist

      self.lastAction = bestAction
      return bestAction

    self.lastAction = action
    return action

  def getQValue(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    qValue = 0.0
    features = self.getFeatures(gameState, action)
    for key in features.keys():
        qValue += (self.weights[key] * features[key])
    return qValue

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    features['successorScore'] = self.getScore(successor)

    foodList = self.getFood(successor).asList()
    myPos = successor.getAgentState(self.index).getPosition()

    # Compute distance to the nearest food
    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
      if self.getFood(gameState)[int(myPos[0])][int(myPos[1])]:
        features['distanceToFood'] = 0

    # # Compute distance to the nearest capsule
    # capsuleList = self.getCapsules(successor)
    # if len(capsuleList) > 0:
    #     minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
    #     features['distanceToCapsule'] = minDistance

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

    # Nearest invader
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
      agentState = successor.getAgentState(self.index)

      # Invader Alert
      if agentState.scaredTimer != 0 and not agentState.isPacman and min(dists) < 5:
          features['invaderAlert'] = 1

    # Nearest Ghost
    ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer == 0 and a.getPosition() != None ]
    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
      if successor.getAgentState(self.index).isPacman and min(dists) < 3:
        features['ghostAlert'] = 1

    features.divideAll(10.0)
    return features

  def getWeights(self):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return self.weights

  def isTraining(self):
    return self.currentEpisode < self.numTraining

  def getBestActionByQValue(self, gameState, actions):
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()

    if len(actions) == 0:
        return Directions.STOP

    qValues = [self.getQValue(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    maxValue = max(qValues)
    bestActions = [a for a, v in zip(actions, qValues) if v == maxValue]
    return random.choice(bestActions)

  def getBestValue(self, state):
    actions = state.getLegalActions(self.index)
    qValues = [self.getQValue(state, a) for a in actions]
    if len(qValues) > 0:
        return max(qValues)
    return 0.0

  def observationFunction(self, gameState):
    if len(self.observationHistory) > 0 and self.isTraining():
      lastState = self.getCurrentObservation()
      reward = self.getReward(gameState, lastState)
      self.episodeRewards += reward
      self.update(lastState, self.lastAction, gameState, reward)

    return gameState.makeObservation(self.index)

  def getReward(self, currentState, lastState):
    reward = 0
    scoreDiff = self.getScore(currentState) - self.getScore(lastState)
    if scoreDiff > 0:
      # print("Won points!!!!!!!!!!!!!!!")
      reward = 100 * scoreDiff

    #FoodEaten
    currentfoodList = self.getFood(currentState).asList()
    lastfoodList = self.getFood(lastState).asList()
    reward += 5*(len(currentfoodList) - len(lastfoodList))

    # if len(currentfoodList) > len(lastfoodList):
    #   print("Eaten food: food count: %d" % len(currentfoodList))
    # elif len(currentfoodList) < len(lastfoodList):
    #   print("food lost: food count: %d" % len(currentfoodList))

    currentDefendFoodList = self.getFoodYouAreDefending(currentState).asList()
    lastDefendFoodList = self.getFoodYouAreDefending(lastState).asList()
    reward -= 5*(len(lastDefendFoodList) - len(currentDefendFoodList))

    # if len(currentDefendFoodList) < len(lastDefendFoodList):
    #   print("Defending food eaten : food count: %d" % len(currentDefendFoodList))
    # elif len(currentDefendFoodList) > len(lastDefendFoodList):
    #   print("Defending food retrieved  : food count: %d" % len(currentDefendFoodList))

    # if len(currentDefendFoodList) - len(lastDefendFoodList) > 0:
    #   print("")

    currentEnemies = [currentState.getAgentState(i) for i in self.getOpponents(currentState)]
    for e in currentEnemies:
      enemyPos = e.getPosition()
      if (int(enemyPos[0]), int(enemyPos[1])) == e.start.pos:
        reward += 10
        # print("Enemy eaten")

    if currentState.getAgentState(self.index).getPosition() == self.start:
      reward -= 30
      # print("Got eaten")

    return reward

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    features = self.getFeatures(state, action)
    diff = self.alpha * ((reward + self.gamma * self.getBestValue(nextState)) - self.getQValue(state, action))
    for feature in features.keys():
      self.weights[feature] = self.weights[feature] + diff * features[feature]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def final(self, state):
    "Called at the end of each game."

    self.observationFunction(state)

    CaptureAgent.final(self, state)
    self.currentEpisode += 1
    print("Episode number: %d" % (self.currentEpisode))
    print("Episode rewards: %d" % (self.episodeRewards))
    self.episodeRewards = 0
    print(self.weights)
    if self.currentEpisode == self.numTraining:
      print
      "FINISHED TRAINING"
      with open('weights' + str(self.index), 'wb') as fp:
          pickle.dump(self.weights, fp)


