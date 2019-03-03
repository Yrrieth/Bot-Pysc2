from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

import numpy
    
# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_SCV_QUICK = actions.FUNCTIONS.Train_SCV_quick.id
_BUILD_SUPPLYDEPOT_SCREEN = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS_SCREEN = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE_QUICK = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_BUILD_REFINERY_SCREEN = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_ENGINEERINGBAY_SCREEN = actions.FUNCTIONS.Build_EngineeringBay_screen.id
_TRAIN_MARAUDER_QUICK = actions.FUNCTIONS.Train_Marauder_quick.id
_BUILD_TECHLAB_QUICK = actions.FUNCTIONS.Build_TechLab_quick.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

# Unit IDs
_PLAYER_SELF = 1

# Features
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Parameters
_NOT_QUEUED = [0]

# python3 -m pysc2.bin.agent --map Simple64 --agent projet7.SimpleAgent --agent_race terran --agent2_race terran 
# \ --difficulty medium --max_episodes 10 --step_mul 20



class SimpleAgent(base_agent.BaseAgent): 

    def reset(self):
      self.scv_selected = False
      self.nb_supply_depot = 0
      self.commandcenter_selected = False
      self.max_supply_depot = 1
      self.barracks_selected = False
      self.state_attack = 0
      self.vespene_location_lock = [0, 0]
      self.scv_selected_for_refinery = False
      self.scv_selected_for_refinery1 = False
      self.barracks_center = 0
      self.techLab_built = False

    """
    Renvoie les coordonées du lieu à attaquer en fonction de la position du joueur
    """
    def coordinate_attack(self, obs):
      player_relative = obs.observation["feature_minimap"][_PLAYER_RELATIVE]
      player_y, player_x = (player_relative == _PLAYER_SELF).nonzero()
      self.top_left_corner = player_y.mean() <= 31
      if self.top_left_corner == True:
        return [42, 45]
      return [18, 22]

    """
    En fonction de la position initiale du joueur, renvoie des coordonnées en fonction du centre du screen
    """
    def change_location(self, obs, coord_x, coord_y):
      player_relative = obs.observation["feature_minimap"][_PLAYER_RELATIVE]
      player_y, player_x = (player_relative == _PLAYER_SELF).nonzero()
      self.top_left_corner = player_y.mean() <= 32 # Fais la moyenne des features en ordonnée du joueur
      if self.top_left_corner == True:             # et si c'est inférieur à 32 (moitié de la minimap), renvoie True
        return [42 + coord_x, 42 + coord_y]
      return [42 - coord_x, 42 - coord_y]

    """
    Sélectionne le centre de commandement puis entraine des SCVs
    """
    def train_SCV(self, obs):
      if self.commandcenter_selected == False :
        if _SELECT_POINT in obs.observation["available_actions"]:
          self.commandcenter_selected = True
          unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
          cc = (unit_type == units.Terran.CommandCenter)
          commandcenter_y, commandcenter_x = cc.nonzero()
          cc_center_x = numpy.mean(commandcenter_x, axis=0).round()
          cc_center_y = numpy.mean(commandcenter_y, axis=0).round()
          cc_target = [cc_center_x, cc_center_y]
          return actions.FUNCTIONS.select_point("select", cc_target)
      if self.commandcenter_selected == True :
        if _TRAIN_SCV_QUICK in obs.observation["available_actions"]:
          return actions.FUNCTIONS.Train_SCV_quick("now")
      return actions.FUNCTIONS.no_op()

    """
    Sélectionne un SCV sur le screen
    """
    def select_scv(self, obs):
      if _SELECT_POINT in obs.observation["available_actions"]:
        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
        scv_y, scv_x = (unit_type == units.Terran.SCV).nonzero()
        target = [scv_x[0], scv_y[0]]
        self.scv_selected = True
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    """
    En fonction de la position du joueur, renvoie la localisation d'un gaz de vespène
    """
    def vespene_location(self, obs):
      if self.top_left_corner == True:
        return [60, 20]
      return [20, 60]

    """
    Construit un dépôt de ravitaillement
    """
    def build_supply_depot(self, obs, coord):
      if _BUILD_SUPPLYDEPOT_SCREEN in obs.observation["available_actions"]:
        target = coord
        return actions.FunctionCall(_BUILD_SUPPLYDEPOT_SCREEN, [_NOT_QUEUED, target])
      return actions.FUNCTIONS.no_op()

    """
    Construit une caserne
    """
    def build_barracks(self, obs, coord):
      if _BUILD_BARRACKS_SCREEN in obs.observation["available_actions"]:
        target = coord
        return actions.FunctionCall(_BUILD_BARRACKS_SCREEN, [_NOT_QUEUED, target])
      return actions.FUNCTIONS.no_op()

    """
    Construit une rafinerie
    """
    def build_refinery(self, obs):
      if _BUILD_REFINERY_SCREEN in obs.observation["available_actions"]:
        target = self.vespene_location(obs)
        return actions.FunctionCall(_BUILD_REFINERY_SCREEN, [_NOT_QUEUED, target])

    """
    Construit un centre technique
    """
    def build_engineeringBay(self, obs, coord):
      if _BUILD_BARRACKS_SCREEN in obs.observation["available_actions"]:
        target = coord
        return actions.FunctionCall(_BUILD_ENGINEERINGBAY_SCREEN, [_NOT_QUEUED, target])
      return actions.FUNCTIONS.no_op()

    """
    Ajoute un SCV à la collecte de vespène
    """
    def add_SCV_harvest_refinery(self, obs):
      target = self.vespene_location(obs)
      return actions.FunctionCall(_HARVEST_GATHER_SCREEN, [_NOT_QUEUED, target])

    """
    Sélectionne une caserne
    """
    def select_barracks(self, obs):
      if _SELECT_POINT in obs.observation["available_actions"]:
        self.barracks_selected = True
        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
        b = (unit_type == units.Terran.Barracks)
        barracks_y, barracks_x = b.nonzero()
        b_center_x = numpy.mean(barracks_x, axis=0).round()
        b_center_y = numpy.mean(barracks_y, axis=0).round()
        self.barracks_center = [b_center_x, b_center_y]
        return actions.FUNCTIONS.select_point("select", self.barracks_center)

    """
    Construit un laboratoire technique, nécessaire pour le maraudeur
    """
    def build_techLab(self, obs):
      if _BUILD_TECHLAB_QUICK in obs.observation["available_actions"]:
        target = self.barracks_center
        self.techLab_built = True
        return actions.FunctionCall(_BUILD_TECHLAB_QUICK, [_NOT_QUEUED])
      return actions.FUNCTIONS.no_op()

    """
    Entraîne une unité de combat, ici, marine et maraudeur
    """
    def train_combat_unit(self, obs, combat_unit):
      if self.barracks_selected == False:
        return self.select_barracks(obs)
      else:
        if combat_unit == "marine":
          if _TRAIN_MARINE_QUICK in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE_QUICK, [_NOT_QUEUED])
        elif combat_unit == "marauder":
          if _TRAIN_MARAUDER_QUICK in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARAUDER_QUICK, [_NOT_QUEUED])
      return actions.FUNCTIONS.no_op()

    """
    Sélectionne l'armée puis les envoie en direction du camp ennemi pour les attaquer depuis la minimap
    """
    def attack(self, obs):
      if self.state_attack == 0:
        if _SELECT_ARMY in obs.observation["available_actions"]:
          self.state_attack = 1
          return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        else:
          return actions.FunctionCall(_NOOP, [])
      if(self.state_attack == 1):
        if _ATTACK_MINIMAP in obs.observation["available_actions"]:
          player_relative = obs.observation["feature_screen"][_PLAYER_RELATIVE]
          marines_y, marines_x = (player_relative == _PLAYER_SELF).nonzero()
          if not marines_y.any():
            return actions.FunctionCall(_NOOP, [])
          self.initial_marines_x = numpy.sum(marines_x)/marines_x.size
          self.initial_marines_y = numpy.sum(marines_y)/marines_y.size
          dest = self.coordinate_attack(obs)
          return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, dest])
      return actions.FunctionCall(_NOOP, [])

    def step(self, obs):
      super(SimpleAgent, self).step(obs)
      collected_vespene = obs.observation["score_cumulative"]["collected_vespene"]
      spent_vespene = obs.observation["score_cumulative"]["spent_vespene"]
      food_army = obs.observation["player"]["food_army"]
      food_workers = obs.observation["player"]["food_workers"]
      value_structures = obs.observation["score_cumulative"]["total_value_structures"]

      if value_structures < 500:
        if self.scv_selected == False:
          return self.select_scv(obs)
        else:
          return self.build_supply_depot(obs, self.change_location(obs, 12, 17))

      if value_structures == 500:
        return self.build_supply_depot(obs, self.change_location(obs, 2, 17))

      if value_structures == 600:
        return self.build_supply_depot(obs, self.change_location(obs, 2, 27))

      if value_structures == 700:
        return self.build_barracks(obs, self.change_location(obs, 27, 17))

      if value_structures == 850:
        return self.build_refinery(obs)

      if value_structures == 925 and food_workers < 13:
        self.scv_selected = False
        return self.train_SCV(obs)

      if self.scv_selected == False and self.scv_selected_for_refinery == False:
        return self.select_scv(obs)
      elif self.scv_selected == True and self.scv_selected_for_refinery == False :
        self.scv_selected_for_refinery = True
        self.scv_selected = False
        return self.add_SCV_harvest_refinery(obs)

      if self.scv_selected == False and self.scv_selected_for_refinery1 == False :
        return self.select_scv(obs)
      elif self.scv_selected == True and self.scv_selected_for_refinery1 == False :
        self.scv_selected_for_refinery1 = True
        self.scv_selected = False
        return self.add_SCV_harvest_refinery(obs)
      
      if value_structures == 925:
        return self.build_engineeringBay(obs, self.change_location(obs, 27, 0))

      if value_structures == 1050 and self.techLab_built == False and self.barracks_selected == False:
        return self.select_barracks(obs)
      elif value_structures == 1050 and self.techLab_built == False and self.barracks_selected == True:
        return self.build_techLab(obs)

      if value_structures >= 1125:
        if food_army > 8:
          self.barracks_selected = False
          return self.attack(obs)
        else:
          self.state_attack = 0
          #if (collected_vespene - spent_vespene) > 25:
            #return self.train_combat_unit(obs, "marauder")
          return self.train_combat_unit(obs, "marine")

      return actions.FUNCTIONS.no_op()
