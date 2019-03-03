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
_BUILD_FACTORY_SCREEN = actions.FUNCTIONS.Build_Factory_screen.id
_TRAIN_HELLION_QUICK = actions.FUNCTIONS.Train_Hellion_quick.id
_TRAIN_WIDOWMINE_QUICK = actions.FUNCTIONS.Train_WidowMine_quick.id
_TRAIN_THOR_QUICK = actions.FUNCTIONS.Train_Thor_quick.id
_BUILD_ARMORY_SCREEN = actions.FUNCTIONS.Build_Armory_screen.id

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
      self.scv_selected_for_refinery_1 = False
      self.scv_selected_for_refinery_2 = False
      self.barracks_center = 0
      self.techLab_built = False
      self.factory_selected = False

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

    def select_CC(self, obs):
      if _SELECT_POINT in obs.observation["available_actions"]:
          self.commandcenter_selected = True
          unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
          cc = (unit_type == units.Terran.CommandCenter)
          commandcenter_y, commandcenter_x = cc.nonzero()
          cc_center_x = numpy.mean(commandcenter_x, axis=0).round()
          cc_center_y = numpy.mean(commandcenter_y, axis=0).round()
          cc_target = [cc_center_x, cc_center_y]
          return actions.FUNCTIONS.select_point("select", cc_target)

    """
    Sélectionne le centre de commandement puis entraine des SCVs
    """
    def train_SCV(self, obs):
      if self.commandcenter_selected == False :
        return self.select_CC(obs)
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
    En fonction de la position du joueur, renvoie la localisation d'un minerai
    """
    def mineral_location(self, obs):
      if self.top_left_corner == True:
        return [40, 20]
      return [20, 40]

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
    Ajoute un SCV à la collecte de minerai
    """
    def add_SCV_harvest_mineral(self, obs):
      target = self.mineral_location(obs)
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
    Construit une usine
    """
    def build_factory(self, obs, coord):
      if _BUILD_FACTORY_SCREEN in obs.observation["available_actions"]:
        return actions.FunctionCall(_BUILD_FACTORY_SCREEN, [_NOT_QUEUED, coord])

    """
    Sélectionne une usine
    """
    def select_factory(self, obs):
      if _SELECT_POINT in obs.observation["available_actions"]:
        self.factory_selected = True
        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
        f = (unit_type == units.Terran.Factory)
        factory_y, factory_x = f.nonzero()
        f_center_x = numpy.mean(factory_x, axis=0).round()
        f_center_y = numpy.mean(factory_y, axis=0).round()
        self.factory_center = [f_center_x, f_center_y]
        return actions.FUNCTIONS.select_point("select", self.factory_center)

    """
    Construit une armurerie
    """
    def build_armory(self, obs, coord):
      if _BUILD_ARMORY_SCREEN in obs.observation["available_actions"]:
        return actions.FunctionCall(_BUILD_ARMORY_SCREEN, [_NOT_QUEUED, coord])

    """
    Entraîne une unité de combat dans une caserne, ici, marine et maraudeur
    """
    def train_combat_unit_in_barracks(self, obs, combat_unit):
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

    def train_combat_unit_in_factory(self, obs, combat_unit):
      if self.factory_selected == False:
        return self.select_factory(obs)
      else:
        if combat_unit == "hellion":
          if _TRAIN_HELLION_QUICK in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_HELLION_QUICK, [_NOT_QUEUED])
        elif combat_unit == "widowMine":
          if _TRAIN_WIDOWMINE_QUICK in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_WIDOWMINE_QUICK, [_NOT_QUEUED])
        elif combat_unit == "thor":
          if _TRAIN_WIDOWMINE_QUICK in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_THOR_QUICK, [_NOT_QUEUED])
      return actions.FUNCTIONS.no_op()

    """
    Sélectionne l'armée
    """
    def select_army(self, obs):
      if _SELECT_ARMY in obs.observation["available_actions"]:
        return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
      else:
        return actions.FunctionCall(_NOOP, [])

    """
    Sélectionne l'armée puis les envoie en direction du camp ennemi pour les attaquer depuis la minimap
    """
    def attack(self, obs):
      if self.state_attack == 0:
        self.state_attack = 1
        return self.select_army(obs)
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

      #print("value_structures : ", value_structures)
      #print("Valeur SCV : ", food_workers, " valeur armee : ", food_army)
      #print(self.scv_selected)

      if value_structures < 500:
        if self.scv_selected == False:
          return self.select_scv(obs)
        else:
          return self.build_supply_depot(obs, self.change_location(obs, 12, 17))

      if value_structures == 500:
        return self.build_supply_depot(obs, self.change_location(obs, 2, 17))

      if value_structures == 600:
        return self.build_refinery(obs)

      if self.commandcenter_selected == False: # On sélectionne le CC pour déselectionner le SCV
        self.scv_selected = False
        return self.select_CC(obs)

      if self.scv_selected == False and self.scv_selected_for_refinery == False:
        return self.select_scv(obs)
      elif self.scv_selected == True and self.scv_selected_for_refinery == False :
        self.scv_selected_for_refinery = True
        self.scv_selected = False
        return self.add_SCV_harvest_refinery(obs)

      if self.scv_selected == False and self.scv_selected_for_refinery_1 == False :
        return self.select_scv(obs)
      elif self.scv_selected == True and self.scv_selected_for_refinery_1 == False :
        self.scv_selected_for_refinery_1 = True
        self.scv_selected = False
        self.commandcenter_selected = False
        return self.add_SCV_harvest_refinery(obs)

      if value_structures == 675:
        if self.scv_selected == False:
          return self.select_scv(obs)
        else:
          return self.build_barracks(obs, self.change_location(obs, 27, 17))

      if value_structures == 825 and self.scv_selected == True and self.scv_selected_for_refinery_2 == False:
        self.scv_selected_for_refinery_2 = True
        self.scv_selected = False
        self.commandcenter_selected = False
        return self.add_SCV_harvest_refinery(obs)

      if value_structures == 825 and food_workers < 13:
        return self.train_SCV(obs)

      if value_structures == 825 and food_army < 2:
        return self.train_combat_unit_in_barracks(obs, "marine")

      if value_structures == 825 and food_army >= 2:
        if self.scv_selected == False:
          return self.select_scv(obs)
        else:
          return self.build_supply_depot(obs, self.change_location(obs, 2, 27))

      #if value_structures == 925 and self.food_army > 10:
          #return self.attack(obs)

      if value_structures == 925:
        return self.build_factory(obs, self.change_location(obs, 27, 2))

      if value_structures == 1175 and food_army < 10: # Une usine coûte 250 de minerai ???
        self.scv_selected = False
        return self.train_combat_unit_in_factory(obs, "hellion")

      if value_structures == 1175 and food_army >= 10:
        if self.scv_selected == False:
          return self.select_scv(obs)
        else:
          self.factory_selected = False
          return self.build_supply_depot(obs, self.change_location(obs, 12, 27))

      if value_structures == 1275:
        return self.build_armory(obs, self.change_location(obs, -13, 27))

      if value_structures == 1525 and food_army >= 10: # Une armurerie coûte 250 de minerai
        if self.factory_selected == False:
          return self.select_factory(obs)
        if self.techLab_built == False and self.factory_selected == True:
          return self.build_techLab(obs)

      if value_structures >= 1575:
        if food_army > 14:
          self.factory_selected = False
          self.barracks_selected = False
          return self.attack(obs)
        else:
          self.state_attack = 0
          if (collected_vespene - spent_vespene) > 200:
            return self.train_combat_unit_in_factory(obs, "thor")
          return self.train_combat_unit_in_factory(obs, "hellion")

      """
      
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
            #return self.train_combat_unit_in_barracks(obs, "marauder")
          return self.train_combat_unit_in_barracks(obs, "marine")
"""
      return actions.FUNCTIONS.no_op()
