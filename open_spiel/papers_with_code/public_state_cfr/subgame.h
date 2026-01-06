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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/data_structures.h"
#include "open_spiel/algorithms/poker_data.h"

// Depth-limited CFR is conceptually similar to what `infostate_cfr.h` does.
// Additionally it saves structure of public states in the root and leaves
// of the depth-limited infostate tree. It collects these into a set of
// `PublicState`s. DL-CFR can be instantiated with custom state evaluators,
// which may need to save different structures associated to these states.
//
// Therefore the state evaluator provides these two methods:
//
// - CreateContext: takes a public state and return any special context
//   representation needed for this state evaluator. These will be saved by
//   DL-CFR to provide them later as necessary.
// - EvaluatePublicState: receives a pointer to the public state along with
//   a context. In the state the evaluator can find the current beliefs, and it
//   should update the counterfactual values to continue the DL-CFR iterations.
//
// DL-CFR can be queried for `Policy` only in the depth-limited constructed
// parts of the tree.

namespace open_spiel {
namespace papers_with_code {

// -- Public state -------------------------------------------------------------

enum PublicStateType {
  // Initial public state of the depth-limited lookahead tree. There might be
  // multiple initial states (i.e. the lookahead tree is more precisely a forest
  // of trees).
  kInitialPublicState,
  // Leaf public state of the depth-limited lookahead tree. All such public
  // states are at the end of the lookahead. Some of them might be terminal,
  // i.e. they contain only `open_spiel::State`s that are terminal.
  kLeafPublicState,
};

struct PublicState {
  // An identification of the public state: a tensor of perfect recall
  // public observation.
  const Observation public_tensor;
  // Public state type. Note that only the pair of (public_tensor, state_type)
  // uniquely identifies public_id: this is for technical reasons so that we can
  // build 1-step lookaheads. There would be a single public state that is both
  // initial and leaf. We need to distinguish between the two, so we save them
  // redundantly as two different public states. The distinction is needed
  // because while the public observations are the same, the infostate node
  // pointers are not, and so are not all the other members derived from the
  // infostate nodes (beliefs and values).
  const PublicStateType state_type;
  // Position in the vector of Subgame::public_states()
  const size_t public_id;
  // TODO
  /*const*/ std::vector<std::shared_ptr<algorithms::InfostateTree>> trees;
  // For each player, store a pointer to the infostate nodes for this public
  // state, within the depth-limited infostate tree. If needed, you can get
  // access to underlying perfect-information `State`s
  // via `InfostateNode::corresponding_states()`.
  // These are assigned once, and no subsequent change should be made.
  // The top nodes are saved for initial states and bottom nodes are saved for
  // leaf states. Both are listed for the special case if the public tree is
  // a singleton.
  /*const*/ std::array<std::vector<const algorithms::InfostateNode*>, 2> nodes;
  // Store a map between infostate nodes and positions in the vectors
  // of beliefs and values. The map is shared across the players.
  /*const*/ std::map<const algorithms::InfostateNode*, int> nodes_positions;
  // Store the move number associated with all the states that belong to this
  // public state.
  /*const*/ int move_number = -1;
  // For each player, store beliefs for the top-most infostate nodes, which
  // correspond to bottom_nodes of the DL infostate tree.
  std::array<std::vector<double>, 2> beliefs;
  // For each player, store current counterfactual values for the top-most
  // infostate nodes, which correspond to bottom_nodes of the DL infostate tree.
  std::array<std::vector<double>, 2> values;
  // For each player, store average counterfactual values for the same
  // infostates as the current cf. values.
  std::array<std::vector<double>, 2> average_values;

  PublicState(const Observation& public_observation,
              const PublicStateType state_type, const size_t public_id);
  // Check if the public state if initial, i.e. the depth-limited subgame
  // is rooted in this public state.
  bool IsInitial() const { return state_type == kInitialPublicState; }
  // Check if the public state if a leaf within the depth-limited subgame.
  bool IsLeaf() const { return state_type == kLeafPublicState; }
  // Check if the public state is terminal, i.e. it contains only states
  // that satisfy `State::IsTerminal()`.
  bool IsTerminal() const;
  // Compute the reach probability of this public state.
  double ReachProbability() const;
  // Check the state has non-zero reach probability.
  bool IsReachable() const { return ReachProbability() > 0; }
  // Check the state has non-zero beliefs for given player.
  bool IsReachableByPlayer(int player) const;
  // Check the state has non-zero beliefs for some player.
  bool IsReachableBySomePlayer() const {
    return IsReachableByPlayer(0) || IsReachableByPlayer(1);
  };
  // Compute value of this state based on current policy (for the given player).
  double CurrentValue(int player = 0) const;
  // Compute value of this state based on average policy (for the given player).
  double AverageValue(int player = 0) const;
  // Check if the public state is zero-sum.
  bool IsZeroSum() const {
    return fabs(CurrentValue(0) + CurrentValue(1)) < 1e-10;
  }
  // Set new beliefs and check they have consistent sizes.
  void SetBeliefs(const std::array<std::vector<double>, 2>& new_beliefs);
  // Return a map of infostate string: average cf. values.
  std::unordered_map<std::string, double> InfostateAvgValues(
      Player player) const;
};

void DebugPrintPublicFeatures(const std::vector<PublicState>& states);

// -- Many public states -------------------------------------------------------

// TODO: merge with Subgame and introduce public tree structure.
//  This struct is pretty much the same, but subgame stores only
//  the initial / leaf public states.
// Store all public states in the game. This requires building the game tree
// for the entire game. All the public states are marked as initial (even
// the terminal ones).
struct PublicStatesInGame {
  std::vector<std::shared_ptr<algorithms::InfostateTree>> infostate_trees;
  std::vector<PublicState> public_states;
  PublicState* GetPublicState(const Observation& public_observation);
};
std::unique_ptr<PublicStatesInGame> MakeAllPublicStates(const Game& game);

// -- Subgame ------------------------------------------------------------------

// Depth-limited CFR requires storage of perfect-information states both
// in the roots and in the leaves of the infostate trees.
constexpr int kDlCfrInfostateTreeStorage = algorithms::kStoreStatesInRoots
                                         | algorithms::kStoreStatesInLeaves;

// Subgame stores initial and leaf public states, associated
// with depth-limited infostate trees.
struct Subgame {
  std::shared_ptr<const Game> game;
  std::shared_ptr<Observer> public_observer;
  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees;
  std::vector<PublicState> public_states;
  std::unordered_map<std::string, int> public_state_id_map;

  Subgame(std::shared_ptr<const Game> game, int max_moves);
  Subgame(std::shared_ptr<const Game> game,
          std::shared_ptr<Observer> public_observer,
          std::vector<std::shared_ptr<algorithms::InfostateTree>> trees);

  PublicState& initial_state() {
    SPIEL_CHECK_FALSE(public_states.empty());
    SPIEL_CHECK_TRUE(public_states[0].IsInitial());
    return public_states[0];
  }

  PublicState* PickRandomLeaf(std::mt19937& engine);
 private:
  void MakePublicStates();
  void MakeBeliefsAndValues();
  PublicState* GetPublicState(const std::string &public_observation_string, const Observation& public_observation,
                              PublicStateType state_type);
  PublicState* GetPublicState(Observation& public_observation,
                              PublicStateType state_type,
                              algorithms::InfostateNode* node);
};

std::unique_ptr<Subgame> MakeSubgame(
    const PublicState& state,
    std::shared_ptr<const Game> game = nullptr,
    std::shared_ptr<Observer> public_observer = nullptr,
    int custom_move_ahead_limit = algorithms::kNoMoveAheadLimit);

// Derived classes specify members as they need for their specific
// public state evaluators.
//
// It is **strongly advised** to make the contexts immutable, or if the state
// evaluator mutates the context for the purposes of its computation, it should
// restore it back to the original. This is because it is beneficial to share
// the same context between multiple evaluator implementations (for example
// the sequence-form oracle evaluators). If mutation were to happen, each
// evaluator then must take this into account, making implementation more
// difficult.
struct PublicStateContext {
  // Make sure PublicStateContext is polymorphic.
  virtual ~PublicStateContext() = default;
};

// Public state evaluator can create appropriate contexts for later evaluation
// of public states. The derived classes should down_cast the context as needed.
// Beliefs and values are saved within the public state.
int ConvertToFullPokerCard(int card, const algorithms::PokerData &poker_data);
bool CompatibleHands(const std::vector<int> &hand_one, const std::vector<int> &hand_two);

class PublicStateEvaluator {
 public:
  virtual ~PublicStateEvaluator() = default;
  virtual std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState &state) const { return nullptr; };
  virtual void ResetContext(PublicStateContext *context) const {}
  virtual void EvaluatePublicState(
      PublicState *public_state, PublicStateContext *context) const = 0;
};

// River network evaluator
struct RiverNetworkPublicStateContext final : public PublicStateContext {
  int pot_;
  int card_;
  explicit RiverNetworkPublicStateContext(const PublicState &state);
};

// General poker terminal evaluator
struct GeneralPokerTerminalFullBoardCardsContext {
  algorithms::PokerData poker_data_;
  int belief_size_;
  std::vector<std::vector<int>> ordered_hands_;
  std::unordered_map<int, std::vector<int>> card_to_possible_hands_;
  std::vector<int> hand_strengths_;
  explicit GeneralPokerTerminalFullBoardCardsContext(const PublicState &state);
};

struct GeneralPokerTerminalPublicStateContext final : public PublicStateContext {
  bool fold_state_;
  std::vector<double> utilities_;
  std::string board_cards_string_;
  explicit GeneralPokerTerminalPublicStateContext(const PublicState &state);
};

struct GoofspielPublicStateContext final : public PublicStateContext {
  std::vector<std::vector<double>> utilities_;
  std::vector<std::vector<std::vector<int>>> belief_map_;
  explicit GoofspielPublicStateContext(const PublicState &state);
};

struct LiarsDicePublicStateContext final : public PublicStateContext {
  std::pair<int, int> bid_;
  std::vector<double> utilities_;
  std::vector<std::vector<int>> face_groups_;
  explicit LiarsDicePublicStateContext(const PublicState &state);
};

class GeneralPokerTerminalEvaluatorLinear : public PublicStateEvaluator {
 public:
  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState &state) const override;
  void EvaluatePublicState(
      PublicState *state, PublicStateContext *context) const override;
 private:
  std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<GeneralPokerTerminalFullBoardCardsContext>>>
      contexts = std::make_shared<std::unordered_map<std::string, std::shared_ptr<GeneralPokerTerminalFullBoardCardsContext>>>();
};

// Terminal evaluator for poker for river subgame
struct PokerTerminalPublicStateContext final : public PublicStateContext {
  bool fold_state_;
  std::vector<double> utilities_;
  explicit PokerTerminalPublicStateContext(const PublicState &state);
};

class PokerTerminalEvaluatorQuadratic final : public PublicStateEvaluator {
 public:
  PokerTerminalEvaluatorQuadratic(algorithms::PokerData poker_data, std::vector<int> board_cards);

  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void EvaluatePublicState(
      PublicState* state, PublicStateContext* context) const override;
 private:
  algorithms::PokerData poker_data_;
  std::unordered_map<int, std::vector<std::vector<int>>> lost_won_mapping_;
  std::unordered_map<int, int> belief_sizes_;
};

class PokerTerminalEvaluatorLinear final : public PublicStateEvaluator {
 public:
  PokerTerminalEvaluatorLinear(algorithms::PokerData poker_data, std::vector<int> board_cards);

  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void EvaluatePublicState(
      PublicState* state, PublicStateContext* context) const override;
 private:
  algorithms::PokerData poker_data_;
  int belief_size_;
  std::vector<std::vector<int>> ordered_hands_;
  std::unordered_map<int, std::vector<int>> card_to_possible_hands_;
  std::vector<int> hand_strengths_;
};

class GoofspielTerminalEvaluator final : public PublicStateEvaluator {
  public:
   std::unique_ptr<PublicStateContext> CreateContext(
       const PublicState& state) const override;
   void EvaluatePublicState(
       PublicState* state, PublicStateContext* context) const override;
  private:
 };

 class LiarsDiceTerminalEvaluator final : public PublicStateEvaluator {
  public:
   std::unique_ptr<PublicStateContext> CreateContext(
       const PublicState& state) const override;
   void EvaluatePublicState(
       PublicState* state, PublicStateContext* context) const override;
  private:
 };

// -- Terminal evaluator -------------------------------------------------------

struct TerminalPublicStateContext final : public PublicStateContext {
  // Map from player 0 index (key) to player 1 (value).
  std::vector<int> permutation;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<double> utilities;
  explicit TerminalPublicStateContext(const PublicState& state);
};

class TerminalEvaluator final : public PublicStateEvaluator {
 public:
  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void EvaluatePublicState(
      PublicState* state, PublicStateContext* context) const override;
};

std::shared_ptr<PublicStateEvaluator> MakePokerTerminalEvaluator(algorithms::PokerData poker_data, std::vector<int> cards);
std::shared_ptr<PublicStateEvaluator> MakeGoofspielTerminalEvaluator();
std::shared_ptr<PublicStateEvaluator> MakeLiarsDiceTerminalEvaluator();
std::shared_ptr<PublicStateEvaluator> MakeTerminalEvaluator();
std::shared_ptr<PublicStateEvaluator> MakeDummyEvaluator();
// CFR evaluator that makes a large number of iterations.
std::shared_ptr<PublicStateEvaluator> MakeApproxOracleEvaluator(
    std::shared_ptr<const Game> game, int cfr_iterations = 100000);

// -- Subgame solver -----------------------------------------------------------

using PolicySelection = algorithms::PolicySelection;
PolicySelection GetSaveValuesPolicy(const std::string& s);
constexpr PolicySelection kDefaultPolicySelection =
    PolicySelection::kAveragePolicy;

// CFR-based subgame solver that evaluates public leaves using terminal
// or non-terminal evaluator.
double RootCfValue(int root_branching_factor,
                   absl::Span<const double> cf_values,
                   absl::Span<const double> range = {});

class SubgameSolver {
 public:
  SubgameSolver(
      std::shared_ptr<Subgame> subgame,
      const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
      const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
      const std::shared_ptr<std::mt19937> rnd_gen,
      const std::string& bandit_name,
      PolicySelection save_values_policy = kDefaultPolicySelection,
      bool safe_resolving = false,
      bool beliefs_for_average = false,
      double noisy_values = 0.);

  void RunSimultaneousIterations(int iterations, bool network_evaluation = false);
  void Reset();

  // Accessors.
  PublicState& initial_state() { return subgame_->initial_state(); }
  std::vector<algorithms::BanditVector>& bandits() { return bandits_; }
  Subgame* subgame() { return subgame_.get(); }

  // Policy available only for the infostates of the subgame!
  std::shared_ptr<Policy> AveragePolicy();
  std::shared_ptr<Policy> CurrentPolicy();
  std::vector<std::vector<double>> GetCfValues() const { return cf_values_;};
  std::vector<double> RootValues() const;
  std::vector<std::vector<double>> GetReaches() const {return reach_probs_;};
 private:
  const std::shared_ptr<Subgame> subgame_;
  const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator_;
  const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator_;
  const std::shared_ptr<std::mt19937> rnd_gen_;
  const bool safe_resolving_;
  const bool beliefs_for_average_;
  const double noisy_values_;

  // -- Mutable values to keep track of. --
  // These have the size at largest depth of the tree, i.e. the size of the
  // leaf infostate nodes.
  std::vector<algorithms::BanditVector> bandits_;
  std::vector<std::vector<double>> reach_probs_;
  std::vector<std::vector<double>> cf_values_;
  // Save evaluator-specific information for any public state.
  // If no information should be saved, a nullptr is used.
  std::vector<std::unique_ptr<PublicStateContext>> contexts_;

  size_t num_iterations_ = 0;
  PolicySelection init_save_values_;

  void EvaluateLeaves();
  void EvaluateLeavesNetwork();
  void EvaluateLeaf(PublicState* state, PublicStateContext* context);
  void CopyCurrentValuesToInitialState();
  void IncrementallyAverageValuesInState(PublicState* state);
};

// -- Dummy evaluator ----------------------------------------------------------

// Evaluator that does nothing.
struct DummyEvaluator : public PublicStateEvaluator {
  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState &state) const override { return nullptr; };
  void ResetContext(PublicStateContext *context) const override {};
  void EvaluatePublicState(PublicState *public_state,
                           PublicStateContext *context) const override {};
};


// -- Poker specific CFR evaluator ---------------------------------------------

struct PokerCFRContext : public PublicStateContext {
  std::unique_ptr<SubgameSolver> dlcfr;
  explicit PokerCFRContext(std::unique_ptr<SubgameSolver> d)
      : dlcfr(std::move(d)) {}
};

struct PokerCFREvaluator : public PublicStateEvaluator {
  std::shared_ptr<const Game> game;
  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator;
  std::shared_ptr<Observer> public_observer;
  std::shared_ptr<Observer> infostate_observer;
  bool reset_subgames_on_evaluation = true;
  int num_cfr_iterations;
  std::string bandit_name = "RegretMatchingPlus";
  PolicySelection save_values_policy = kDefaultPolicySelection;

  PokerCFREvaluator(std::shared_ptr<const Game> game,
                    std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
                    std::shared_ptr<Observer> public_observer,
                    std::shared_ptr<Observer> infostate_observer,
                    int cfr_iterations = 100);

  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState &state) const override;
  void ResetContext(PublicStateContext *context) const override;
  void EvaluatePublicState(PublicState *state,
                           PublicStateContext *context) const override;
};

// -- CFR evaluator ------------------------------------------------------------

struct CFRContext : public PublicStateContext {
  std::unique_ptr<SubgameSolver> dlcfr;
  explicit CFRContext(std::unique_ptr<SubgameSolver> d)
      : dlcfr(std::move(d)) {}
};

struct CFREvaluator : public PublicStateEvaluator {
  std::shared_ptr<const Game> game;
  int depth_limit;
  std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator;
  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator;
  std::shared_ptr<Observer> public_observer;
  std::shared_ptr<Observer> infostate_observer;
  bool reset_subgames_on_evaluation = true;
  int num_cfr_iterations;
  std::string bandit_name = "RegretMatchingPlus";
  PolicySelection save_values_policy = kDefaultPolicySelection;

  CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
               std::shared_ptr<const PublicStateEvaluator> leaf_evaluator,
               std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
               std::shared_ptr<Observer> public_observer,
               std::shared_ptr<Observer> infostate_observer,
               int cfr_iterations = 100);

  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void ResetContext(PublicStateContext* context) const override;
  void EvaluatePublicState(PublicState* state,
                           PublicStateContext* context) const override;
};

void PrintPublicStatesStats(const std::vector<PublicState>& public_leaves);

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_
