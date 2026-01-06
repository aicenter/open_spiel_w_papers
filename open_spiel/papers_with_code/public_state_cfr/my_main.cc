// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "algorithms/best_response.h"
#include "algorithms/cfr.h"
#include "algorithms/expected_returns.h"
#include "algorithms/tabular_exploitability.h"
#include "infostate_tree_br.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/algorithms/poker_data.h"
#include "policy.h"
#include "subgame.h"

#include <iostream>
#include <fstream>

std::string POKER_GAME_NAME = "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,firstPlayer=2 1 1 1,"
                     "numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=20000 20000,bettingAbstraction=fcpa)";
std::vector<int> CARD_INDICES = {4, 31, 10, 15, 20};
std::string BOARD_CARDS = "9s7c5s4h3c";
int POT_SIZE = 200;

namespace open_spiel {
namespace papers_with_code {
namespace {
void UpdateChanceReaches(std::vector<double> &chance_reaches,
                         const algorithms::PokerData &poker_data,
                         const std::vector<int> &cards) {
  chance_reaches;
  for (int card : cards) {
    for (int hand_index : poker_data.card_to_hands_.at(card)) {
      chance_reaches[hand_index] = 0;
    }
  }
  double mag = 0;
  for (double mag_part : chance_reaches) {
    mag += mag_part;
  }
  for (double &chance_reach : chance_reaches) {
    chance_reach /= mag;
  }
}

std::pair<std::shared_ptr<Subgame>, algorithms::PokerData> MakeSubgame(std::string name, std::vector<int> cards) {
  std::shared_ptr<const Game> game = LoadGame(name);

  std::unique_ptr<State> state = game->NewInitialState();

  // Deal 4 cards
  state->ApplyAction(0);
  state->ApplyAction(1);
  state->ApplyAction(2);
  state->ApplyAction(3);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal 3 board cards (Flop)
  state->ApplyAction(cards[0]);
  state->ApplyAction(cards[1]);
  state->ApplyAction(cards[2]);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (Turn)
  state->ApplyAction(cards[3]);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (River)
  state->ApplyAction(cards[4]);

  std::cout << state->ToString() << "\n";

  universal_poker::logic::CardSet card_set(cards);
  std::cout << card_set.ToString() << "\n";

  std::shared_ptr<Observer> infostate_observer =
  game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer =
  game->MakeObserver(kPublicStateObsType, {});

  std::vector<double> chance_reaches(1326, 1. / 1326);

  algorithms::PokerData poker_data = algorithms::PokerData(*state);

  UpdateChanceReaches(chance_reaches, poker_data, cards);

  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees =
  algorithms::MakePokerInfostateTrees(state, chance_reaches,
                        infostate_observer, 1000,
                        kDlCfrInfostateTreeStorage, cards);

  auto out = std::make_shared<Subgame>(game, public_observer, trees);

  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
  std::make_shared<const PokerTerminalEvaluatorQuadratic>(poker_data,
                                            cards);

  SubgameSolver solver =
  SubgameSolver(out, nullptr, terminal_evaluator,
  std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  return std::pair<std::shared_ptr<Subgame>, algorithms::PokerData>{out, poker_data};
}

std::pair<int, int> UniversalPokerRiverCFRPokerSpecificLinear(int iterations, bool run_exploitability) {
  std::string name = POKER_GAME_NAME;

  std::vector<int> cards = CARD_INDICES;

  auto start = std::chrono::high_resolution_clock::now();

  auto [out, poker_data] = MakeSubgame(name, cards);

  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      std::make_shared<const PokerTerminalEvaluatorLinear>(poker_data, cards);

  SubgameSolver solver =
      SubgameSolver(out, nullptr, terminal_evaluator,
                    std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (run_exploitability) {
    auto fixed_policy = solver.AveragePolicy();

    std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(),
                                                      TabularPolicy()};

    for (int player = 0; player < 2; player++) {
      algorithms::BanditVector &bandits = solver.bandits()[player];
      for (algorithms::DecisionId id : bandits.range()) {
        algorithms::InfostateNode *node =
            solver.subgame()->trees[player]->decision_infostate(id);
        const std::string &infostate = node->infostate_string();
        ActionsAndProbs infostate_policy =
            fixed_policy->GetStatePolicy(infostate);
        separated_policies[player].SetStatePolicy(infostate, infostate_policy);
      }
    }

    double nash_conv = 0;

    for (int player = 0; player < 2; player++) {
      SubgameSolver best_response =
          SubgameSolver(out, nullptr, terminal_evaluator,
                        std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
      best_response.bandits() =
          MakeResponseBandits(out->trees, separated_policies[1 - player]);
      best_response.RunSimultaneousIterations(1);
      std::cout << best_response.RootValues() << "\n";
      nash_conv += best_response.RootValues()[player];
    }
    std::cout << "Exploitability: " << nash_conv / 2 << "\n";
  }
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int>
UniversalPokerRiverCFRPokerSpecificQuadratic(int iterations, bool run_exploitability) {
  std::string name = POKER_GAME_NAME;
  std::vector<int> cards = CARD_INDICES;

  auto start = std::chrono::high_resolution_clock::now();
  
  auto [out, poker_data] = MakeSubgame(name, cards);

  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      std::make_shared<const PokerTerminalEvaluatorQuadratic>(poker_data,
                                                              cards);

  SubgameSolver solver =
      SubgameSolver(out, nullptr, terminal_evaluator,
                    std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (run_exploitability) {
    auto fixed_policy = solver.AveragePolicy();

    std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(),
                                                        TabularPolicy()};

    for (int player = 0; player < 2; player++) {
      algorithms::BanditVector &bandits = solver.bandits()[player];
      for (algorithms::DecisionId id : bandits.range()) {
        algorithms::InfostateNode *node =
            solver.subgame()->trees[player]->decision_infostate(id);
        const std::string &infostate = node->infostate_string();
        ActionsAndProbs infostate_policy =
            fixed_policy->GetStatePolicy(infostate);
        separated_policies[player].SetStatePolicy(infostate, infostate_policy);
      }
    }

    double nash_conv = 0;

    for (int player = 0; player < 2; player++) {
      SubgameSolver best_response =
          SubgameSolver(out, nullptr, terminal_evaluator,
                        std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
      best_response.bandits() =
          MakeResponseBandits(out->trees, separated_policies[1 - player]);
      best_response.RunSimultaneousIterations(1);
      std::cout << best_response.RootValues() << "\n";
      nash_conv += best_response.RootValues()[player];
    }
    std::cout << "Exploitability: " << nash_conv / 2 << "\n";
  }
      
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> UniversalPokerRiverCFREfg(int iterations, bool run_exploitability) {
  std::mt19937 rng(0);

  int pot_size = POT_SIZE;
  std::string board_cards = BOARD_CARDS;

  std::vector<double> uniform_reaches;
  uniform_reaches.reserve(2 * universal_poker::kSubgameUniqueHands);
  for (int i = 0; i < 2 * universal_poker::kSubgameUniqueHands; ++i) {
    uniform_reaches.push_back(1. / (2 * universal_poker::kSubgameUniqueHands));
  }
  std::shared_ptr<const Game> game = universal_poker::MakeRandomSubgame(
      rng, pot_size, board_cards, uniform_reaches);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::CFRSolverBase solver(*game, false, false, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (run_exploitability) {
    std::string name = POKER_GAME_NAME;
    std::vector<int> cards = CARD_INDICES;
    
    auto [out, poker_data] = MakeSubgame(name, cards);
    
    std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      std::make_shared<const PokerTerminalEvaluatorLinear>(poker_data,
                                                              cards);

    SubgameSolver subgame_solver =
      SubgameSolver(out, nullptr, terminal_evaluator,
                    std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

    auto fixed_policy = solver.AveragePolicy();

    std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(),
                                                        TabularPolicy()};

    for (int player = 0; player < 2; player++) {
      algorithms::BanditVector &bandits = subgame_solver.bandits()[player];
      for (algorithms::DecisionId id : bandits.range()) {
        algorithms::InfostateNode *node =
            subgame_solver.subgame()->trees[player]->decision_infostate(id);
        const std::string &infostate = node->infostate_string();
        if(solver.InfoStateValuesTable().find(infostate) != solver.InfoStateValuesTable().end()) {
          std::cout << "setting policy" << std::endl;
          ActionsAndProbs infostate_policy =
              fixed_policy->GetStatePolicy(infostate);
          separated_policies[player].SetStatePolicy(infostate, infostate_policy);
        }
      }
    }

    double nash_conv = 0;

    for (int player = 0; player < 2; player++) {
      SubgameSolver best_response =
          SubgameSolver(out, nullptr, terminal_evaluator,
                        std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
      best_response.bandits() =
          MakeResponseBandits(out->trees, separated_policies[1 - player]);
      best_response.RunSimultaneousIterations(1);
      std::cout << best_response.RootValues() << "\n";
      nash_conv += best_response.RootValues()[player];
    }
    std::cout << "Exploitability: " << nash_conv / 2 << "\n";
  }

  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> UniversalPokerRiverCFREfgHashed(int iterations, bool run_exploitability) {
  std::mt19937 rng(0);

  int pot_size = POT_SIZE;
  std::string board_cards = BOARD_CARDS;

  std::vector<double> uniform_reaches;
  uniform_reaches.reserve(2 * universal_poker::kSubgameUniqueHands);
  for (int i = 0; i < 2 * universal_poker::kSubgameUniqueHands; ++i) {
    uniform_reaches.push_back(1. / (2 * universal_poker::kSubgameUniqueHands));
  }
  std::shared_ptr<const Game> game = universal_poker::MakeRandomSubgame(
      rng, pot_size, board_cards, uniform_reaches);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::CFRSolverBase solver(*game, false, false, true, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (run_exploitability) {
    std::string name = POKER_GAME_NAME;
    std::vector<int> cards = CARD_INDICES;
    
    auto [out, poker_data] = MakeSubgame(name, cards);
    
    std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      std::make_shared<const PokerTerminalEvaluatorLinear>(poker_data,
                                                              cards);

    SubgameSolver subgame_solver =
      SubgameSolver(out, nullptr, terminal_evaluator,
                    std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

    auto fixed_policy = solver.AveragePolicy();

    std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(),
                                                        TabularPolicy()};

    for (int player = 0; player < 2; player++) {
      algorithms::BanditVector &bandits = subgame_solver.bandits()[player];
      for (algorithms::DecisionId id : bandits.range()) {
        algorithms::InfostateNode *node =
            subgame_solver.subgame()->trees[player]->decision_infostate(id);
        const std::string &infostate = node->infostate_string();
        if(solver.InfoStateValuesTable().find(infostate) != solver.InfoStateValuesTable().end()) {
          ActionsAndProbs infostate_policy =
              fixed_policy->GetStatePolicy(infostate);
              separated_policies[player].SetStatePolicy(infostate, infostate_policy);
        }
      }
    }

    double nash_conv = 0;

    for (int player = 0; player < 2; player++) {
      SubgameSolver best_response =
          SubgameSolver(out, nullptr, terminal_evaluator,
                        std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
      best_response.bandits() =
          MakeResponseBandits(out->trees, separated_policies[1 - player]);
      best_response.RunSimultaneousIterations(1);
      std::cout << best_response.RootValues() << "\n";
      nash_conv += best_response.RootValues()[player];
    }
    std::cout << "Exploitability: " << nash_conv / 2 << "\n";
  }

  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> LiarsDiceOpenSpielGamePscfr(int iterations,
                                                std::string game_name, bool run_exploitability) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  auto start = std::chrono::high_resolution_clock::now();
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});

  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees =
      algorithms::MakeLiarsDiceInfostateTrees(game->NewInitialState(), 1.0,
                                              infostate_observer, 1000000,
                                              kDlCfrInfostateTreeStorage);

  auto out = std::make_shared<Subgame>(game, public_observer, trees);

  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      MakeLiarsDiceTerminalEvaluator();

  SubgameSolver solver =
      SubgameSolver(out, nullptr, terminal_evaluator,
                    std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  auto fixed_policy = solver.AveragePolicy();

  if (run_exploitability) {
    double exploitability = algorithms::Exploitability(*game, *fixed_policy);
    std::cout << "Exploitability: " << exploitability << "\n";
  }

  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> OpenSpielGamePscfrVanilla(int iterations,
                                                       std::string game_name, bool run_exploitability) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  auto start = std::chrono::high_resolution_clock::now();
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});

  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees =
      algorithms::MakeInfostateTrees(*game, 1000000,
                                     kDlCfrInfostateTreeStorage);

  auto out = std::make_shared<Subgame>(game, public_observer, trees);

  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();

  SubgameSolver solver =
      SubgameSolver(out, nullptr, terminal_evaluator,
                    std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  auto fixed_policy = solver.AveragePolicy();

  if (run_exploitability) {
    double exploitability = algorithms::Exploitability(*game, *fixed_policy);
    std::cout << "Exploitability: " << exploitability << "\n";
  }

  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> GoofspielOpenSpielGamePscfr(int iterations,
                                                std::string game_name, bool run_exploitability) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  auto start = std::chrono::high_resolution_clock::now();
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});

  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees =
      algorithms::MakeEfficientInfostateTrees(game->NewInitialState(), 1.0,
                                              infostate_observer, 1000000,
                                              kDlCfrInfostateTreeStorage);

  auto out = std::make_shared<Subgame>(game, public_observer, trees);

  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      MakeGoofspielTerminalEvaluator();

  SubgameSolver solver =
      SubgameSolver(out, nullptr, terminal_evaluator,
                    std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  // solver.RunSimultaneousIterations(iterations);
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  auto fixed_policy = solver.AveragePolicy();


  if (run_exploitability) {
    double exploitability = algorithms::Exploitability(*game, *fixed_policy);
    std::cout << "Exploitability: " << exploitability << "\n";
  }

  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> GeneralGameCFREfg(int iterations, std::string game_name, bool run_exploitability) {
  std::mt19937 rng(0);

  std::shared_ptr<const Game> game = LoadGame(game_name);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::CFRSolverBase solver(*game, false, true, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (run_exploitability) {
    double exploitability =
        algorithms::Exploitability(*game, *solver.AveragePolicy());
    std::cout << "Exploitability: " << exploitability << "\n";
  }

    
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> GeneralGameCFREfgHashed(int iterations,
                                            std::string game_name, bool run_exploitability) {
  std::mt19937 rng(0);

  std::shared_ptr<const Game> game = LoadGame(game_name);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::CFRSolverBase solver(*game, false, true, true, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  end = std::chrono::high_resolution_clock::now();
  auto run_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (run_exploitability) {
    double exploitability =
        algorithms::Exploitability(*game, *solver.AveragePolicy());
    std::cout << "Exploitability: " << exploitability << "\n";
  }

  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

void MeasureTime(int runs, int iterations, bool run_exploitability, std::pair<int, int> (*f)(int, bool)) {
  std::vector<std::pair<int, int>> collected_times;
  collected_times.reserve(runs);
  for (int i = 0; i < runs; i++) {
    collected_times.push_back(f(iterations, run_exploitability));
  }
  std::pair<int, int> cumulative_times(0, 0);
  for (auto &time_pair : collected_times) {
    cumulative_times.first += time_pair.first;
    cumulative_times.second += time_pair.second;
  }
  std::cout << "Average setup time: " << cumulative_times.first / runs
            << "ms\n";
  std::cout << "Average runin time: " << cumulative_times.second / runs
            << "ms\n";
}

void MeasureTime(int runs, int iterations, std::string game_name,
                 bool run_exploitability, std::pair<int, int> (*f)(int, std::string, bool)) {
  std::vector<std::pair<int, int>> collected_times;
  collected_times.reserve(runs);
  for (int i = 0; i < runs; i++) {
    collected_times.push_back(f(iterations, game_name, run_exploitability));
  }
  std::pair<int, int> cumulative_times(0, 0);
  for (auto &time_pair : collected_times) {
    cumulative_times.first += time_pair.first;
    cumulative_times.second += time_pair.second;
  }
  std::cout << "Average setup time: " << cumulative_times.first / runs
            << "ms\n";
  std::cout << "Average runin time: " << cumulative_times.second / runs
            << "ms\n";
}
} // unnamed namespace
} // namespace papers_with_code
} // namespace open_spiel

ABSL_FLAG(int, iterations, 1000, "How many iterations to run.");
ABSL_FLAG(int, runs, 1, "How many runs to run.");
ABSL_FLAG(std::string, game_name, "", "The name of the game to run the algorithm on.");
ABSL_FLAG(bool, pscfr_linear, false, "Whether to run PSCFR with linear evaluator.");
ABSL_FLAG(bool, pscfr_quadratic, false, "Whether to run PSCFR with quadratic evaluator.");
ABSL_FLAG(bool, efgcfr, false, "Whether to run EFG-CFR.");
ABSL_FLAG(bool, efgcfr_cached, false, "Whether to run EFG-CFR with cached structure.");
ABSL_FLAG(bool, exploitability, false, "Whether to run exploitability.");


int main(int argc, char **argv) {

  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);  
  
  std::string game_name = absl::GetFlag(FLAGS_game_name);

  int iterations = absl::GetFlag(FLAGS_iterations);
  int runs = absl::GetFlag(FLAGS_runs);

  // Create the game.
  bool run_pscfr_linear = absl::GetFlag(FLAGS_pscfr_linear);
  bool run_pscfr_quadratic = absl::GetFlag(FLAGS_pscfr_quadratic);
  bool run_efgcfr_normal = absl::GetFlag(FLAGS_efgcfr);
  bool run_efgcfr_cached = absl::GetFlag(FLAGS_efgcfr_cached);
  bool run_exploitability = absl::GetFlag(FLAGS_exploitability);

  std::cout << "Running " << game_name << " with " << iterations << " iterations and " << runs << " runs" << std::endl;
  std::cout << "Running PSCFR with linear evaluator: " << run_pscfr_linear << std::endl;
  std::cout << "Running PSCFR with quadratic evaluator: " << run_pscfr_quadratic << std::endl;
  std::cout << "Running EFG-CFR: " << run_efgcfr_normal << std::endl;
  std::cout << "Running EFG-CFR with cached structure: " << run_efgcfr_cached << std::endl;
  std::cout << "Running exploitability: " << run_exploitability << std::endl;

  std::string game_class;

  // Check if arguments include name of a game
  if (game_name.substr(0, 9) == "goofspiel" && game_name.length() > 9) {
    std::string num_cards =
    game_name.substr(9); // Extract number after "goofspiel"
    game_name = "turn_based_simultaneous_game(game=goofspiel(imp_info="
                "True,num_cards=" +
                num_cards + ",points_order=descending))";
    game_class = "goofspiel";
  } else if (game_name.substr(0, 10) == "liars_dice" && game_name.length() > 10) {
    game_class = "liars_dice";
  }
  if (game_name == "poker") {
    // Linear evaluator infostate CFR
    if (run_pscfr_linear) {
      std::cout << "Infostate CFR experiment with Linear evaluator:"
                << std::endl;
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, run_exploitability,
          open_spiel::papers_with_code::
              UniversalPokerRiverCFRPokerSpecificLinear);
    }
    // Quadratic evaluator infostate CFR
    if (run_pscfr_quadratic) {
      std::cout << "Infostate CFR experiment with Quadratic evaluator:"
                << std::endl;
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, run_exploitability,
          open_spiel::papers_with_code::
              UniversalPokerRiverCFRPokerSpecificQuadratic);
    }
    // Efg CFR without saving structure
    if (run_efgcfr_normal) {
      std::cout << "EFG CFR experiment:" << std::endl;
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, run_exploitability,
          open_spiel::papers_with_code::UniversalPokerRiverCFREfg);
    }
    // Efg CFR with saving the structure
    if (run_efgcfr_cached) {
      std::cout << "Hashed EFG CFR experiment:" << std::endl;
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, run_exploitability,
          open_spiel::papers_with_code::UniversalPokerRiverCFREfgHashed);
    }
  } else {
    // PSCFR since for general game we have no specific terminal evaluator
    if (run_pscfr_linear) {
      std::cout << "PSCFR on " << game_name << std::endl;
      if (game_class == "goofspiel") {
        open_spiel::papers_with_code::MeasureTime(
            runs, iterations, game_name, run_exploitability,
            open_spiel::papers_with_code::GoofspielOpenSpielGamePscfr);
      } else if (game_class == "liars_dice") {
          open_spiel::papers_with_code::MeasureTime(
              runs, iterations, game_name, run_exploitability,
              open_spiel::papers_with_code::LiarsDiceOpenSpielGamePscfr);
      }
    }
    if(run_pscfr_quadratic) {
      std::cout << "PSCFR quadratic on " << game_name << std::endl;
      open_spiel::papers_with_code::MeasureTime(
        runs, iterations, game_name, run_exploitability,
        open_spiel::papers_with_code::
            OpenSpielGamePscfrVanilla);
    }
    // Efg CFR without saving structure
    if (run_efgcfr_normal) {
      std::cout << "EFGCFR on " << game_name << std::endl;
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, game_name, run_exploitability,
          open_spiel::papers_with_code::GeneralGameCFREfg);
    }
    // Efg CFR with saving the structure
    if (run_efgcfr_cached) {
      std::cout << "EFGCFR cached on " << game_name << std::endl;
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, game_name, run_exploitability,
          open_spiel::papers_with_code::GeneralGameCFREfgHashed);
    }
  }
}