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

#include "open_spiel/algorithms/infostate_tree.h"

#include <iomanip>
#include <limits>
#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "games/liars_dice.h"
#include "open_spiel/action_view.h"

namespace open_spiel {
namespace algorithms {

using internal::kUndefinedNodeId;

InfostateNode::InfostateNode(
    const InfostateTree &tree, InfostateNode *parent, int incoming_index,
    InfostateNodeType type, const std::string &infostate_string,
    double terminal_utility, double terminal_ch_reach_prob, size_t depth,
    std::vector<Action> legal_actions, std::vector<Action> terminal_history,
    std::unordered_set<std::string> opponent_infostate_strings)
    : tree_(tree), parent_(parent), incoming_index_(incoming_index),
      type_(type), infostate_string_(infostate_string),
      terminal_utility_(terminal_utility),
      terminal_chn_reach_prob_(terminal_ch_reach_prob), depth_(depth),
      legal_actions_(std::move(legal_actions)),
      terminal_history_(std::move(terminal_history)),
      opponent_infostate_strings_(std::move(opponent_infostate_strings)) {
  // Implications for kTerminalNode
  SPIEL_CHECK_TRUE(type_ != kTerminalInfostateNode || parent_);
  SPIEL_CHECK_TRUE(type_ != kTerminalInfostateNode ||
                   std::isfinite(terminal_utility));
  SPIEL_CHECK_TRUE(type_ != kTerminalInfostateNode ||
                   (std::isfinite(terminal_ch_reach_prob) &&
                    terminal_ch_reach_prob >= 0 &&
                    terminal_ch_reach_prob <= 1));
  // Implications for kDecisionNode
  SPIEL_CHECK_TRUE(type_ != kDecisionInfostateNode || parent_);
  // Implications for kObservationNode
  SPIEL_CHECK_TRUE(!(type_ == kObservationInfostateNode && parent_ &&
                     parent_->type() == kDecisionInfostateNode) ||
                   (incoming_index_ >= 0 &&
                    incoming_index_ < parent_->legal_actions().size()));
}

InfostateNode *InfostateNode::AddChild(std::unique_ptr<InfostateNode> child) {
  SPIEL_CHECK_EQ(child->parent_, this);
  children_.push_back(std::move(child));
  return children_.back().get();
}

InfostateNode *
InfostateNode::GetChild(const std::string &infostate_string) const {
  for (const std::unique_ptr<InfostateNode> &child : children_) {
    if (child->infostate_string() == infostate_string)
      return child.get();
  }
  return nullptr;
}

std::vector<InfostateNode *> InfostateNode::children() const {
  std::vector<InfostateNode *> out;
  out.reserve(children_.size());
  for (const std::unique_ptr<InfostateNode> &child : children_) {
    out.push_back(child.get());
  }
  return out;
}

std::ostream &InfostateNode::operator<<(std::ostream &os) const {
  if (!parent_)
    return os << 'x';
  return os << parent_ << ',' << incoming_index_;
}

std::string InfostateNode::TreePath() const {
  if (!parent_)
    return "x";
  return absl::StrCat(parent_->TreePath(), ",", incoming_index_);
}

std::string InfostateNode::MakeCertificate(int precision) const {
  if (type_ == kTerminalInfostateNode) {
    if (precision > -1) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(precision) << terminal_utility_;
      std::string mystring = ss.str();
      return "{" + ss.str() + "}";
    } else {
      return "{}";
    }
  }

  std::vector<std::string> certificates;
  for (InfostateNode *child : child_iterator()) {
    certificates.push_back(child->MakeCertificate(precision));
  }
  std::sort(certificates.begin(), certificates.end());

  std::string open, close;
  if (type_ == kDecisionInfostateNode) {
    open = "[";
    close = "]";
  } else if (type_ == kObservationInfostateNode) {
    open = "(";
    close = ")";
  }
  return absl::StrCat(
      open, absl::StrJoin(certificates.begin(), certificates.end(), ""), close);
}

std::string InfostateNode::MakeCertificate() const {
  return this->MakeCertificate(-1);
}

void InfostateNode::RebalanceSubtree(int target_depth, int current_depth) {
  SPIEL_DCHECK_LE(current_depth, target_depth);
  depth_ = current_depth;

  if (is_leaf_node() && target_depth != current_depth) {
    // Prepare the chain of dummy observations.
    depth_ = target_depth;
    std::unique_ptr<InfostateNode> node = Release();
    InfostateNode *node_parent = node->parent();
    int position_in_leaf_parent = node->incoming_index();
    std::unique_ptr<InfostateNode> chain_head =
        std::unique_ptr<InfostateNode>(new InfostateNode(
            /*tree=*/tree_, /*parent=*/nullptr,
            /*incoming_index=*/position_in_leaf_parent,
            kObservationInfostateNode,
            /*infostate_string=*/kFillerInfostate,
            /*terminal_utility=*/NAN, /*terminal_ch_reach_prob=*/NAN,
            current_depth, /*legal_actions=*/{}, /*terminal_history=*/{}));
    InfostateNode *chain_tail = chain_head.get();
    for (int i = 1; i < target_depth - current_depth; ++i) {
      chain_tail =
          chain_tail->AddChild(std::unique_ptr<InfostateNode>(new InfostateNode(
              /*tree=*/tree_, /*parent=*/chain_tail,
              /*incoming_index=*/0, kObservationInfostateNode,
              /*infostate_string=*/kFillerInfostate,
              /*terminal_utility=*/NAN, /*terminal_ch_reach_prob=*/NAN,
              current_depth + i, /*legal_actions=*/{},
              /*terminal_history=*/{})));
    }
    chain_tail->children_.push_back(nullptr);

    // First put the node to the chain. If we did it in reverse order,
    // i.e chain to parent and then node to the chain, the node would
    // become freed.
    auto *node_ptr = node.get();
    node_ptr->SwapParent(std::move(node), /*target=*/chain_tail, 0);
    auto *chain_head_ptr = chain_head.get();
    chain_head_ptr->SwapParent(std::move(chain_head), /*target=*/node_parent,
                               position_in_leaf_parent);
  }

  for (std::unique_ptr<InfostateNode> &child : children_) {
    child->RebalanceSubtree(target_depth, current_depth + 1);
  }
}

std::unique_ptr<InfostateNode> InfostateNode::Release() {
  SPIEL_DCHECK_TRUE(parent_);
  SPIEL_DCHECK_TRUE(parent_->children_.at(incoming_index_).get() == this);
  return std::move(parent_->children_.at(incoming_index_));
}

void InfostateNode::SwapParent(std::unique_ptr<InfostateNode> self,
                               InfostateNode *target, int at_index) {
  // This node is still who it thinks it is :)
  SPIEL_DCHECK_TRUE(self.get() == this);
  target->children_.at(at_index) = std::move(self);
  this->parent_ = target;
  this->incoming_index_ = at_index;
}

void InfostateTree::RebalanceTree() {
  root_->RebalanceSubtree(tree_height(), 0);
}

void InfostateTree::CollectNodesAtDepth(InfostateNode *node, size_t depth) {
  nodes_at_depths_[depth].push_back(node);
  for (InfostateNode *child : node->child_iterator())
    CollectNodesAtDepth(child, depth + 1);
}

std::ostream &InfostateTree::operator<<(std::ostream &os) const {
  return os << "Infostate tree for player " << acting_player_ << ".\n"
            << "Tree height: " << tree_height_ << '\n'
            << "Root branching: " << root_branching_factor() << '\n'
            << "Number of decision infostate nodes: " << num_decisions() << '\n'
            << "Number of sequences: " << num_sequences() << '\n'
            << "Number of leaves: " << num_leaves() << '\n'
            << "Tree certificate: " << '\n'
            << root().MakeCertificate() << '\n';
}

std::unique_ptr<InfostateNode> InfostateTree::MakeNode(
    InfostateNode *parent, InfostateNodeType type,
    const std::string &infostate_string, double terminal_utility,
    double terminal_ch_reach_prob, size_t depth, const State *originating_state,
    const std::vector<Action> &given_legal_actions, bool terminate,
    const std::vector<std::string> &opponent_infostate_strings) {

  std::vector<Action> legal_actions;
  if (given_legal_actions.empty()) {
    legal_actions =
        originating_state && originating_state->IsPlayerActing(acting_player_)
            ? originating_state->LegalActions(acting_player_)
            : std::vector<Action>();
  } else {
    legal_actions = given_legal_actions;
  }
  std::vector<Action> terminal_history;
  if (is_resolving_tree_) {
    terminal_history = originating_state ? originating_state->History()
                                         : std::vector<Action>();
    if (terminate) {
      terminal_history.push_back(Action(-1));
    }
  } else {
    terminal_history = originating_state && originating_state->IsTerminal()
                           ? originating_state->History()
                           : std::vector<Action>();
  }

  // Instantiate node using new to make sure that we can call
  // the private constructor.
  auto node = std::unique_ptr<InfostateNode>(new InfostateNode(
      *this, parent, parent->num_children(), type, infostate_string,
      terminal_utility, terminal_ch_reach_prob, depth, std::move(legal_actions),
      std::move(terminal_history)));
  return node;
}

std::unique_ptr<InfostateNode> InfostateTree::MakeRootNode() const {
  return std::unique_ptr<InfostateNode>(new InfostateNode(
      /*tree=*/*this, /*parent=*/nullptr, /*incoming_index=*/0,
      /*type=*/kObservationInfostateNode,
      /*infostate_string=*/kDummyRootNodeInfostate,
      /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN,
      /*depth=*/0, /*legal_actions=*/{}, /*terminal_history=*/{}));
}

void InfostateTree::UpdateLeafNode(size_t leaf_depth) {
  tree_height_ = std::max(tree_height_, leaf_depth);
}

void InfostateTree::AddCorrespondingState(InfostateNode *node,
                                          const State &state,
                                          double chance_reach_probs) {
  if (!storage_policy_)
    return;
  bool should_store = false;
  const bool is_terminal = node->type_ == kTerminalInfostateNode;
  const bool is_leaf = state.MoveNumber() >= move_limit_ || is_terminal;
  const bool is_root_child = node->is_root_child();
  const bool is_body_node = !is_leaf && !is_root_child;
  if (storage_policy_ & kStoreStatesInTerminals)
    should_store |= is_terminal;
  if (storage_policy_ & kStoreStatesInLeaves)
    should_store |= is_leaf;
  if (storage_policy_ & kStoreStatesInRoots)
    should_store |= is_root_child;
  if (storage_policy_ & kStoreStatesInBody)
    should_store |= is_body_node;

  if (should_store) {
    node->corresponding_states_.push_back(state.Clone());
    node->corresponding_ch_reaches_.push_back(chance_reach_probs);
  }
}

void InfostateTree::AddPokerCorrespondingState(InfostateNode *node,
                                               const State &state,
                                               double chance_reach_probs) {
  node->corresponding_states_.push_back(state.Clone());
  node->corresponding_ch_reaches_.push_back(chance_reach_probs);
}

std::pair<std::string, std::string>
InfostateTree::ExtractInfostateString(const std::string &infostate_string,
                                      const PokerData &poker_data) {
  return {infostate_string.substr(0, 13),
          infostate_string.substr(24 + poker_data.cards_in_hand_ * 2)};
}

int ConvertToFullPokerCard(int card, const algorithms::PokerData &poker_data) {
  int rank = (int)(card / poker_data.num_suits_);
  int suit = card % poker_data.num_suits_;
  return rank * 4 + suit;
}

std::string InfostateTree::ConstructInfostateString(
    const std::pair<std::string, std::string> &parts,
    const std::vector<int> &card_vector,
    const algorithms::PokerData &poker_data) {
  std::vector<int> converted_cards(card_vector.size());
  for (int i = 0; i < card_vector.size(); i++) {
    converted_cards[i] = ConvertToFullPokerCard(card_vector[i], poker_data);
  }
  universal_poker::logic::CardSet cards(converted_cards);
  return parts.first + "[Private: " + cards.ToString() + "]" + parts.second;
}

void InfostateTree::RecursivelyBuildPokerTree(
    const std::vector<InfostateNode *> &parents, size_t depth,
    const State &state, const std::vector<double> &chance_reach_probs,
    const PokerData &poker_data, int round,
    const std::vector<int> &board_cards) {
  // If we are building safe resolving trees, we have to add additional
  // nodes before going into the actual nodes.
  if (is_resolving_tree_ and depth == 1) {
    SpielFatalError("Not implemented.");
  }
  if (state.IsTerminal()) {
    BuildTerminalPokerNodes(parents, depth, state, chance_reach_probs,
                            poker_data, round);
  } else if (state.IsPlayerActing(acting_player_)) {
    BuildDecisionPokerNodes(parents, depth, state, chance_reach_probs,
                            poker_data, round, board_cards);
  } else {
    BuildObservationPokerNode(parents, depth, state, chance_reach_probs,
                              poker_data, round, board_cards);
  }
}

void InfostateTree::BuildTerminalPokerNodes(
    const std::vector<InfostateNode *> &parents, size_t depth,
    const State &state, const std::vector<double> &chance_reach_probs,
    const PokerData &poker_data, int round) {
  // Get poker state
  auto poker_state =
      down_cast<const open_spiel::universal_poker::UniversalPokerState &>(
          state);

  // Set terminal utility based on call or
  int acting_player_ante = poker_state.acpc_state().Ante(acting_player_);
  int opponent_ante = poker_state.acpc_state().Ante(1 - acting_player_);
  int terminal_utility = acting_player_ante;
  if (acting_player_ante > opponent_ante) {
    terminal_utility = opponent_ante;
  } else if (acting_player_ante < opponent_ante) {
    terminal_utility = -acting_player_ante;
  }

  std::pair<std::string, std::string> parts = ExtractInfostateString(
      infostate_observer_->StringFrom(state, acting_player_), poker_data);

  for (int hand_index = 0; hand_index < poker_data.num_hands_; hand_index++) {
    InfostateNode *node = parents[hand_index]->AddChild(MakeNode(
        parents[hand_index], kTerminalInfostateNode,
        ConstructInfostateString(
            parts, poker_data.hand_to_cards_.at(hand_index), poker_data),
        terminal_utility, chance_reach_probs[hand_index], depth, &state));
    AddPokerCorrespondingState(node, state, chance_reach_probs[hand_index]);
  }

  UpdateLeafNode(depth);
}

void InfostateTree::BuildDecisionPokerNodes(
    const std::vector<InfostateNode *> &parents, size_t depth,
    const State &state, const std::vector<double> &chance_reach_probs,
    const PokerData &poker_data, int round,
    const std::vector<int> &board_cards) {

  // For debugging
  const bool is_leaf_node = round >= move_limit_;
  std::pair<std::string, std::string> parts = ExtractInfostateString(
      infostate_observer_->StringFrom(state, acting_player_), poker_data);

  std::vector<InfostateNode *> new_parents;
  new_parents.reserve(poker_data.num_hands_);

  for (int hand_index = 0; hand_index < poker_data.num_hands_; hand_index++) {
    new_parents.push_back(parents[hand_index]->AddChild(MakeNode(
        parents[hand_index], kDecisionInfostateNode,
        ConstructInfostateString(
            parts, poker_data.hand_to_cards_.at(hand_index), poker_data),
        /*terminal_utility=*/NAN, chance_reach_probs[hand_index], depth,
        &state)));
    AddPokerCorrespondingState(new_parents.back(), state,
                               chance_reach_probs[hand_index]);
  }
  if (is_leaf_node) {
    UpdateLeafNode(depth);
    return;
  }
  for (Action action : state.LegalActions()) {
    RecursivelyBuildPokerTree(new_parents, depth + 1, *state.Child(action),
                              chance_reach_probs, poker_data, round,
                              board_cards);
  }
}

std::vector<double>
UpdateChanceReaches(const std::vector<double> &chance_reaches,
                    const PokerData &poker_data, int card,
                    const std::vector<int> &board_cards) {
  std::vector<double> new_chance_reaches = chance_reaches;
  for (int hand_index : poker_data.card_to_hands_.at(card)) {
    new_chance_reaches[hand_index] = 0;
  }
  auto div = double(poker_data.num_cards_ - board_cards.size() -
                    poker_data.cards_in_hand_);
  for (double &new_chance_reach : new_chance_reaches) {
    new_chance_reach /= div;
  }
  return new_chance_reaches;
}

void InfostateTree::BuildObservationPokerNode(
    const std::vector<InfostateNode *> &parents, size_t depth,
    const State &state, const std::vector<double> &chance_reach_probs,
    const PokerData &poker_data, int round,
    const std::vector<int> &board_cards) {
  SPIEL_DCHECK_TRUE(state.IsChanceNode() ||
                    !state.IsPlayerActing(acting_player_));
  const bool is_leaf_node = round >= move_limit_;

  std::pair<std::string, std::string> parts = ExtractInfostateString(
      infostate_observer_->StringFrom(state, acting_player_), poker_data);

  std::vector<InfostateNode *> new_parents;
  new_parents.reserve(poker_data.num_hands_);

  for (int hand_index = 0; hand_index < poker_data.num_hands_; hand_index++) {
    new_parents.push_back(parents[hand_index]->AddChild(MakeNode(
        parents[hand_index], kObservationInfostateNode,
        ConstructInfostateString(
            parts, poker_data.hand_to_cards_.at(hand_index), poker_data),
        /*terminal_utility=*/NAN, chance_reach_probs[hand_index], depth,
        &state)));
    AddPokerCorrespondingState(new_parents.back(), state,
                               chance_reach_probs[hand_index]);
  }
  if (is_leaf_node) {
    UpdateLeafNode(depth);
    return; // Do not build deeper.
  }
  if (state.IsChanceNode()) {
    for (int next_card = 0; next_card < poker_data.num_cards_; next_card++) {
      if (std::find(board_cards.begin(), board_cards.end(), next_card) ==
          board_cards.end()) {
        std::unique_ptr<State> child = state.Child(next_card);
        std::vector<int> new_board_cards = board_cards;
        new_board_cards.push_back(next_card);
        std::vector<double> new_chance_reaches = UpdateChanceReaches(
            chance_reach_probs, poker_data, next_card, board_cards);
        RecursivelyBuildPokerTree(new_parents, depth + 1, *child,
                                  new_chance_reaches, poker_data, round + 1,
                                  new_board_cards);
      }
    }
  } else {
    for (Action a : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(a);
      RecursivelyBuildPokerTree(new_parents, depth + 1, *child,
                                chance_reach_probs, poker_data, round,
                                board_cards);
    }
  }
}

void InfostateTree::RecursivelyBuildEfficientTree(
    InfostateNode *parent, size_t depth, const State &state,
    double chance_reach_prob, std::string last_infostate,
    std::string last_infostate_opponent) {
  // If we are building safe resolving trees, we have to add additional
  // nodes before going into the actual nodes.
  if (is_resolving_tree_ and depth == 1) {
    SpielFatalError("Not implemented.");
  }
  if (state.IsTerminal()) {
    BuildTerminalEfficientNode(parent, depth, state, chance_reach_prob,
                               last_infostate, last_infostate_opponent);
  } else if (state.IsPlayerActing(acting_player_)) {
    BuildDecisionEfficientNode(parent, depth, state, chance_reach_prob,
                               last_infostate, last_infostate_opponent);
  } else {
    BuildObservationEfficientNode(parent, depth, state, chance_reach_prob,
                                  last_infostate, last_infostate_opponent);
  }
}

void InfostateTree::BuildTerminalEfficientNode(
    InfostateNode *parent, size_t depth, const State &state,
    double chance_reach_prob, std::string last_infostate,
    std::string last_infostate_opponent) {

  auto terminal_node = parent->GetChild(last_infostate);

  if (!terminal_node) {
    InfostateNode *terminal_node = parent->AddChild(MakeNode(
        parent, kTerminalInfostateNode,
        infostate_observer_->StringFrom(state, acting_player_),
        state.Returns()[acting_player_], chance_reach_prob, depth, &state));
    UpdateLeafNode(depth);
    AddCorrespondingState(terminal_node, state, chance_reach_prob);
    terminal_node->add_opponent_infostate_string(last_infostate_opponent);
  } else {
    terminal_node->increment_terminal_chance_reach_prob(chance_reach_prob);
    SPIEL_CHECK_EQ(state.Returns()[acting_player_],
                   terminal_node->terminal_utility());
    terminal_node->add_opponent_infostate_string(last_infostate_opponent);
  }
}

void InfostateTree::BuildDecisionEfficientNode(
    InfostateNode *parent, size_t depth, const State &state,
    double chance_reach_prob, std::string last_infostate,
    std::string last_infostate_opponent) {
  // SPIEL_DCHECK_EQ(parent->type(), kObservationInfostateNode);
  std::string info_state =
      infostate_observer_->StringFrom(state, acting_player_);
  std::string info_state_opponent =
      infostate_observer_->StringFrom(state, 1 - acting_player_);
  InfostateNode *decision_node = parent->GetChild(info_state);
  const bool is_leaf_node = state.MoveNumber() >= move_limit_;

  last_infostate = info_state;
  last_infostate_opponent = info_state_opponent;

  if (decision_node) {
    // The decision node has been already constructed along with children
    // for each action: these are observation nodes.
    // Fetches the observation child and goes deeper recursively.
    SPIEL_DCHECK_EQ(decision_node->type(), kDecisionInfostateNode);
    // AddCorrespondingState(decision_node, state, chance_reach_prob);
    if (is_leaf_node) {
      UpdateLeafNode(depth);
      return; // Do not build deeper.
    }

    if (state.IsSimultaneousNode()) {
      const ActionView action_view(state);
      for (int i = 0; i < action_view.legal_actions[acting_player_].size();
           ++i) {
        InfostateNode *observation_node = decision_node->child_at(i);
        SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);

        for (Action flat_actions :
             action_view.fixed_action(acting_player_, i)) {
          std::unique_ptr<State> child = state.Child(flat_actions);
          RecursivelyBuildEfficientTree(observation_node, depth + 2, *child,
                                        chance_reach_prob, last_infostate,
                                        last_infostate_opponent);
        }
      }
    } else {
      std::vector<Action> legal_actions = state.LegalActions(acting_player_);
      for (int i = 0; i < legal_actions.size(); ++i) {
        InfostateNode *observation_node = decision_node->child_at(i);
        SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);
        std::unique_ptr<State> child = state.Child(legal_actions.at(i));
        RecursivelyBuildEfficientTree(observation_node, depth + 2, *child,
                                      chance_reach_prob, last_infostate,
                                      last_infostate_opponent);
      }
    }
  } else { // The decision node was not found yet.
    decision_node = parent->AddChild(MakeNode(
        parent, kDecisionInfostateNode, info_state,
        /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth, &state));
    AddCorrespondingState(decision_node, state, chance_reach_prob);
    if (is_leaf_node) {
      UpdateLeafNode(depth);
      return; // Do not build deeper.
    }

    // Build observation nodes right away after the decision node.
    // This is because the player might be acting multiple times in a row:
    // each time it might get some observations that branch the infostate
    // tree.

    if (state.IsSimultaneousNode()) {
      ActionView action_view(state);
      for (int i = 0; i < action_view.legal_actions[acting_player_].size();
           ++i) {
        // We build a dummy observation node.
        // We can't ask for a proper infostate string or an originating state,
        // because such a thing is not properly defined after only a partial
        // application of actions for the sim move state
        // (We need to supply all the actions).
        InfostateNode *observation_node = decision_node->AddChild(
            MakeNode(decision_node, kObservationInfostateNode,
                     /*infostate_string=*/kFillerInfostate,
                     /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth,
                     /*originating_state=*/nullptr));

        for (Action flat_actions :
             action_view.fixed_action(acting_player_, i)) {
          // Only now we can advance the state, when we have all actions.
          std::unique_ptr<State> child = state.Child(flat_actions);
          RecursivelyBuildEfficientTree(observation_node, depth + 2, *child,
                                        chance_reach_prob, last_infostate,
                                        last_infostate_opponent);
        }
      }
    } else { // Not a sim move node.
      for (Action a : state.LegalActions()) {
        std::unique_ptr<State> child = state.Child(a);
        InfostateNode *observation_node = decision_node->AddChild(
            MakeNode(decision_node, kObservationInfostateNode,
                     infostate_observer_->StringFrom(*child, acting_player_),
                     /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth,
                     child.get()));
        RecursivelyBuildEfficientTree(observation_node, depth + 2, *child,
                                      chance_reach_prob, last_infostate,
                                      last_infostate_opponent);
      }
    }
  }
}

void InfostateTree::BuildObservationEfficientNode(
    InfostateNode *parent, size_t depth, const State &state,
    double chance_reach_prob, std::string last_infostate,
    std::string last_infostate_opponent) {
  SPIEL_DCHECK_TRUE(state.IsChanceNode() ||
                    !state.IsPlayerActing(acting_player_));
  const bool is_leaf_node = state.MoveNumber() >= move_limit_;

  std::string info_state =
      infostate_observer_->StringFrom(state, acting_player_);
  std::string info_state_opponent =
      infostate_observer_->StringFrom(state, 1 - acting_player_);

  last_infostate = info_state;
  last_infostate_opponent = info_state_opponent;

  InfostateNode *observation_node = parent->GetChild(info_state);
  if (!observation_node) {
    observation_node = parent->AddChild(MakeNode(
        parent, kObservationInfostateNode, info_state,
        /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth, &state));
  }
  SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);
  AddCorrespondingState(observation_node, state, chance_reach_prob);
  if (is_leaf_node) {
    UpdateLeafNode(depth);
    return; // Do not build deeper.
  }

  if (state.IsChanceNode()) {
    for (std::pair<Action, double> action_prob : state.ChanceOutcomes()) {
      std::unique_ptr<State> child = state.Child(action_prob.first);
      RecursivelyBuildEfficientTree(observation_node, depth + 1, *child,
                                    chance_reach_prob * action_prob.second,
                                    last_infostate, last_infostate_opponent);
    }
  } else {
    for (Action a : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(a);
      RecursivelyBuildEfficientTree(observation_node, depth + 1, *child,
                                    chance_reach_prob, last_infostate,
                                    last_infostate_opponent);
    }
  }
}

void InfostateTree::RecursivelyBuildLiarsDiceTree(
    const std::vector<InfostateNode *> &parents, size_t depth,
    const State &state, const std::vector<double> &chance_reach_probs,
    int num_dice, int dice_sides) {
  // If we are building safe resolving trees, we have to add additional
  // nodes before going into the actual nodes.
  if (is_resolving_tree_ and depth == 1) {
    SpielFatalError("Not implemented.");
  }
  if (state.IsTerminal()) {
    BuildTerminalLiarsDiceNodes(parents, depth, state, chance_reach_probs,
                                num_dice, dice_sides);
  } else if (state.IsPlayerActing(acting_player_)) {
    BuildDecisionLiarsDiceNodes(parents, depth, state, chance_reach_probs,
                                num_dice, dice_sides);
  } else {
    BuildObservationLiarsDiceNodes(parents, depth, state, chance_reach_probs,
                                   num_dice, dice_sides);
  }
}

std::vector<int> IndexToRoll(int roll_index, int num_dice, int dice_sides) {
  std::vector<int> rolls(num_dice);
  for (int i = 0; i < num_dice; i++) {
    rolls[i] = (roll_index % dice_sides) + 1; // dice values start at 1
    roll_index /= dice_sides;
  }
  return rolls;
}

void InfostateTree::BuildTerminalLiarsDiceNodes(
    const std::vector<InfostateNode *> &parents, size_t depth,
    const State &state, const std::vector<double> &chance_reach_probs,
    int num_dice, int dice_sides) {
  // Get poker state
  auto liars_dice_state =
      down_cast<const open_spiel::liars_dice::LiarsDiceState &>(
          state);

  int utility = liars_dice_state.calling_player() == acting_player_ ? -1 : 1;

  std::string bid_sequence =
      infostate_observer_->StringFrom(state, acting_player_).substr(num_dice);
  int num_possible_rolls = num_dice * dice_sides;

  for (int roll_index = 0; roll_index < num_possible_rolls; roll_index++) {
    InfostateNode *node = parents[roll_index]->AddChild(MakeNode(
        parents[roll_index], kTerminalInfostateNode,
        absl::StrCat(absl::StrJoin(IndexToRoll(roll_index, num_dice, dice_sides), ""), bid_sequence),
        utility, chance_reach_probs[roll_index], depth, &state));
    AddPokerCorrespondingState(node, state, chance_reach_probs[roll_index]);
  }

  UpdateLeafNode(depth);
}

void InfostateTree::BuildDecisionLiarsDiceNodes(
    const std::vector<InfostateNode *> &parents, size_t depth,
    const State &state, const std::vector<double> &chance_reach_probs,
    int num_dice, int dice_sides) {

  std::string bid_sequence =
      infostate_observer_->StringFrom(state, acting_player_).substr(num_dice);

  std::vector<InfostateNode *> new_parents;
  int num_possible_rolls = num_dice * dice_sides;
  new_parents.reserve(num_possible_rolls);

  for (int roll_index = 0; roll_index < num_possible_rolls; roll_index++) {
    std::vector<int> roll = IndexToRoll(roll_index, num_dice, dice_sides);
    new_parents.push_back(parents[roll_index]->AddChild(
        MakeNode(parents[roll_index], kDecisionInfostateNode,
                 absl::StrCat(absl::StrJoin(roll, ""), bid_sequence),
                 /*terminal_utility=*/NAN, chance_reach_probs[roll_index],
                 depth, &state)));
    AddPokerCorrespondingState(new_parents.back(), state,
                               chance_reach_probs[roll_index]);
  }
  for (Action action : state.LegalActions()) {
    RecursivelyBuildLiarsDiceTree(new_parents, depth + 1, *state.Child(action),
                                  chance_reach_probs, num_dice, dice_sides);
  }
}

void InfostateTree::BuildObservationLiarsDiceNodes(
    const std::vector<InfostateNode *> &parents, size_t depth,
    const State &state, const std::vector<double> &chance_reach_probs,
    int num_dice, int dice_sides) {
  SPIEL_DCHECK_TRUE(!state.IsPlayerActing(acting_player_));

  std::string bid_sequence =
      infostate_observer_->StringFrom(state, acting_player_).substr(num_dice);

  int num_possible_rolls = num_dice * dice_sides;
  std::vector<InfostateNode *> new_parents;
  new_parents.reserve(num_possible_rolls);

  for (int roll_index = 0; roll_index < num_possible_rolls; roll_index++) {
    std::vector<int> roll = IndexToRoll(roll_index, num_dice, dice_sides);
    new_parents.push_back(parents[roll_index]->AddChild(
        MakeNode(parents[roll_index], kObservationInfostateNode,
                 absl::StrCat(absl::StrJoin(roll, ""), bid_sequence),
                 /*terminal_utility=*/NAN, chance_reach_probs[roll_index],
                 depth, &state)));
    AddPokerCorrespondingState(new_parents.back(), state,
                               chance_reach_probs[roll_index]);
  }
  for (Action a : state.LegalActions()) {
    std::unique_ptr<State> child = state.Child(a);
    RecursivelyBuildLiarsDiceTree(new_parents, depth + 1, *child,
                                  chance_reach_probs, num_dice, dice_sides);
  }
}

void InfostateTree::RecursivelyBuildTree(InfostateNode *parent, size_t depth,
                                         const State &state,
                                         double chance_reach_prob) {
  //  SPIEL_CHECK_GT(chance_reach_prob, 0);

  // If we are building safe resolving trees, we have to add additional
  // nodes before going into the actual nodes.
  if (is_resolving_tree_ and depth == 1) {
    auto [next_parent, next_depth] =
        BuildResolvingNodes(parent, depth, state, chance_reach_prob);
    parent = next_parent;
    depth = next_depth;
  }

  if (state.IsTerminal()) {
    return BuildTerminalNode(parent, depth, state, chance_reach_prob);
  } else if (state.IsPlayerActing(acting_player_)) {
    return BuildDecisionNode(parent, depth, state, chance_reach_prob);
  } else {
    return BuildObservationNode(parent, depth, state, chance_reach_prob);
  }
}

std::pair<InfostateNode *, int>
InfostateTree::BuildResolvingNodes(InfostateNode *parent, size_t depth,
                                   const State &state,
                                   double chance_reach_prob) {
  SPIEL_DCHECK_EQ(parent->type(), kObservationInfostateNode);

  // Perspectives by either FT player or the acting player.
  const std::string ft_infostate =
      infostate_observer_->StringFrom(state, ft_player_);
  const std::string acting_infostate =
      infostate_observer_->StringFrom(state, acting_player_);
  // ?
  const std::string ft_decision_infostate =
      kFtInfostatePrefix + acting_infostate;
  const std::string before_t_obs_infostate =
      kTerminateObservationPrefix + acting_infostate;

  // Create or lookup the observation node before the terminals.
  InfostateNode *ft_node = parent->GetChild(ft_decision_infostate);
  InfostateNode *t_obs_node = nullptr;
  if (!ft_node) {
    if (ft_player_ == acting_player_) {
      ft_node = parent->AddChild(
          MakeNode(parent, kDecisionInfostateNode, ft_decision_infostate,
                   /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth,
                   &state, {kActionFollow, kActionTerminate}));
    } else {
      ft_node = parent->AddChild(MakeNode(
          parent, kObservationInfostateNode, ft_decision_infostate,
          /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth, &state));
    }
    AddCorrespondingState(ft_node, state, chance_reach_prob);
    parent = ft_node;
    depth++;

    t_obs_node = parent->AddChild(
        MakeNode(parent, kObservationInfostateNode, before_t_obs_infostate,
                 /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth + 1,
                 &state));
    AddCorrespondingState(t_obs_node, state, chance_reach_prob);
  } else {
    AddCorrespondingState(ft_node, state, chance_reach_prob);
    parent = ft_node;
    t_obs_node = parent->GetChild(before_t_obs_infostate);
    AddCorrespondingState(t_obs_node, state, chance_reach_prob);
  }

  SPIEL_CHECK_TRUE(t_obs_node);
  const int num_children = t_obs_node->num_children();
  const std::string terminal_infostate =
      kTerminatePrefix + acting_infostate + std::to_string(num_children);
  const double terminal_utility = (ft_player_ == acting_player_ ? 1 : -1) *
                                  cf_value_constraints_.at(ft_infostate);

  // Add terminal node.
  InfostateNode *terminal_node = t_obs_node->AddChild(MakeNode(
      t_obs_node, kTerminalInfostateNode, terminal_infostate, terminal_utility,
      chance_reach_prob, depth + 2, &state, std::vector<Action>(), true));
  UpdateLeafNode(depth + 2);
  AddCorrespondingState(terminal_node, state, chance_reach_prob);

  return {parent, depth};
}

void InfostateTree::BuildTerminalNode(InfostateNode *parent, size_t depth,
                                      const State &state,
                                      double chance_reach_prob) {
  const double terminal_utility = state.Returns()[acting_player_];
  InfostateNode *terminal_node = parent->AddChild(
      MakeNode(parent, kTerminalInfostateNode,
               infostate_observer_->StringFrom(state, acting_player_),
               terminal_utility, chance_reach_prob, depth, &state));
  UpdateLeafNode(depth);
  AddCorrespondingState(terminal_node, state, chance_reach_prob);
}

void InfostateTree::BuildDecisionNode(InfostateNode *parent, size_t depth,
                                      const State &state,
                                      double chance_reach_prob) {
  // SPIEL_DCHECK_EQ(parent->type(), kObservationInfostateNode);
  std::string info_state =
      infostate_observer_->StringFrom(state, acting_player_);
  InfostateNode *decision_node = parent->GetChild(info_state);
  const bool is_leaf_node = state.MoveNumber() >= move_limit_;

  if (decision_node) {
    // The decision node has been already constructed along with children
    // for each action: these are observation nodes.
    // Fetches the observation child and goes deeper recursively.
    SPIEL_DCHECK_EQ(decision_node->type(), kDecisionInfostateNode);
    // AddCorrespondingState(decision_node, state, chance_reach_prob);
    if (is_leaf_node) {
      UpdateLeafNode(depth);
      return; // Do not build deeper.
    }

    if (state.IsSimultaneousNode()) {
      const ActionView action_view(state);
      for (int i = 0; i < action_view.legal_actions[acting_player_].size();
           ++i) {
        InfostateNode *observation_node = decision_node->child_at(i);
        SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);

        for (Action flat_actions :
             action_view.fixed_action(acting_player_, i)) {
          std::unique_ptr<State> child = state.Child(flat_actions);
          RecursivelyBuildTree(observation_node, depth + 2, *child,
                               chance_reach_prob);
        }
      }
    } else {
      std::vector<Action> legal_actions = state.LegalActions(acting_player_);
      for (int i = 0; i < legal_actions.size(); ++i) {
        InfostateNode *observation_node = decision_node->child_at(i);
        SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);
        std::unique_ptr<State> child = state.Child(legal_actions.at(i));
        RecursivelyBuildTree(observation_node, depth + 2, *child,
                             chance_reach_prob);
      }
    }
  } else { // The decision node was not found yet.
    decision_node = parent->AddChild(MakeNode(
        parent, kDecisionInfostateNode, info_state,
        /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth, &state));
    AddCorrespondingState(decision_node, state, chance_reach_prob);
    if (is_leaf_node) {
      UpdateLeafNode(depth);
      return; // Do not build deeper.
    }

    // Build observation nodes right away after the decision node.
    // This is because the player might be acting multiple times in a row:
    // each time it might get some observations that branch the infostate
    // tree.

    if (state.IsSimultaneousNode()) {
      ActionView action_view(state);
      for (int i = 0; i < action_view.legal_actions[acting_player_].size();
           ++i) {
        // We build a dummy observation node.
        // We can't ask for a proper infostate string or an originating state,
        // because such a thing is not properly defined after only a partial
        // application of actions for the sim move state
        // (We need to supply all the actions).
        InfostateNode *observation_node = decision_node->AddChild(
            MakeNode(decision_node, kObservationInfostateNode,
                     /*infostate_string=*/kFillerInfostate,
                     /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth,
                     /*originating_state=*/nullptr));

        for (Action flat_actions :
             action_view.fixed_action(acting_player_, i)) {
          // Only now we can advance the state, when we have all actions.
          std::unique_ptr<State> child = state.Child(flat_actions);
          RecursivelyBuildTree(observation_node, depth + 2, *child,
                               chance_reach_prob);
        }
      }
    } else { // Not a sim move node.
      for (Action a : state.LegalActions()) {
        std::unique_ptr<State> child = state.Child(a);
        InfostateNode *observation_node = decision_node->AddChild(
            MakeNode(decision_node, kObservationInfostateNode,
                     infostate_observer_->StringFrom(*child, acting_player_),
                     /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth,
                     child.get()));
        RecursivelyBuildTree(observation_node, depth + 2, *child,
                             chance_reach_prob);
      }
    }
  }
}

void InfostateTree::BuildObservationNode(InfostateNode *parent, size_t depth,
                                         const State &state,
                                         double chance_reach_prob) {
  SPIEL_DCHECK_TRUE(state.IsChanceNode() ||
                    !state.IsPlayerActing(acting_player_));
  const bool is_leaf_node = state.MoveNumber() >= move_limit_;
  const std::string info_state =
      infostate_observer_->StringFrom(state, acting_player_);

  InfostateNode *observation_node = parent->GetChild(info_state);
  if (!observation_node) {
    observation_node = parent->AddChild(MakeNode(
        parent, kObservationInfostateNode, info_state,
        /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, depth, &state));
  }
  SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);
  AddCorrespondingState(observation_node, state, chance_reach_prob);
  if (is_leaf_node) {
    UpdateLeafNode(depth);
    return; // Do not build deeper.
  }

  if (state.IsChanceNode()) {
    for (std::pair<Action, double> action_prob : state.ChanceOutcomes()) {
      std::unique_ptr<State> child = state.Child(action_prob.first);
      RecursivelyBuildTree(observation_node, depth + 1, *child,
                           chance_reach_prob * action_prob.second);
    }
  } else {
    for (Action a : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(a);
      RecursivelyBuildTree(observation_node, depth + 1, *child,
                           chance_reach_prob);
    }
  }
}

int InfostateTree::root_branching_factor() const {
  return root_->num_children();
}

std::shared_ptr<InfostateTree> MakeInfostateTree(const Game &game,
                                                 Player acting_player,
                                                 int max_move_ahead_limit,
                                                 int storage_policy) {
  // Uses new instead of make_shared, because shared_ptr is not a friend and
  // can't call private constructors.
  std::unique_ptr<State> s = game.NewInitialState();
  return std::shared_ptr<InfostateTree>(new InfostateTree(
      std::vector<const State *>{s.get()}, /*chance_reach_probs=*/{1.},
      game.MakeObserver(kInfoStateObsType, {}), acting_player,
      max_move_ahead_limit, storage_policy));
}

std::shared_ptr<InfostateTree>
MakeInfostateTree(const std::vector<InfostateNode *> &start_nodes,
                  int max_move_ahead_limit, int storage_policy) {
  std::vector<const InfostateNode *> const_nodes(start_nodes.begin(),
                                                 start_nodes.end());
  return MakeInfostateTree(const_nodes, max_move_ahead_limit, storage_policy);
}

std::shared_ptr<InfostateTree>
MakeInfostateTree(const std::vector<const InfostateNode *> &start_nodes,
                  int max_move_ahead_limit, int storage_policy) {
  SPIEL_CHECK_FALSE(start_nodes.empty());
  const InfostateNode *some_node = start_nodes[0];
  const InfostateTree &originating_tree = some_node->tree();
  SPIEL_DCHECK_TRUE([&]() {
    for (const InfostateNode *node : start_nodes) {
      // The same-depth constraint may be removed in the future in needed.
      // In fact, it is possible to make infostate trees also with non-sibling
      // nodes (i.e. a node can be a descendant of another node in the list).
      // But they should all probably belong to the same tree.
      if (node->depth() != some_node->depth())
        return false;
      if (&node->tree() != &originating_tree)
        return false;
    }
    return true;
  }());

  // We reserve a larger number of states, as infostate nodes typically contain
  // a large number of States. (8 is an arbitrary choice though).
  std::vector<const State *> start_states;
  start_states.reserve(start_nodes.size() * 8);
  std::vector<double> chance_reach_probs;
  chance_reach_probs.reserve(start_nodes.size() * 8);

  for (const InfostateNode *node : start_nodes) {
    SPIEL_CHECK_TRUE(node);
    SPIEL_CHECK_FALSE(node->corresponding_states().empty());
    for (int i = 0; i < node->corresponding_states_size(); ++i) {
      start_states.push_back(node->corresponding_states()[i].get());
      chance_reach_probs.push_back(node->corresponding_chance_reach_probs()[i]);
    }
  }

  // Uses new instead of make_shared, because shared_ptr is not a friend and
  // can't call private constructors.
  return std::shared_ptr<InfostateTree>(new InfostateTree(
      start_states, chance_reach_probs, originating_tree.infostate_observer_,
      originating_tree.acting_player_, max_move_ahead_limit, storage_policy));
}

std::shared_ptr<InfostateTree> MakePokerInfostateTree(
    const std::unique_ptr<State> &start_state,
    const std::vector<double> &chance_reach_probs,
    const std::shared_ptr<Observer> &infostate_observer, Player acting_player,
    const std::vector<int> &board_cards, int round_limit, int storage_policy) {
  return std::shared_ptr<InfostateTree>(new InfostateTree(
      start_state, chance_reach_probs, infostate_observer, acting_player,
      round_limit, storage_policy, board_cards));
}

std::shared_ptr<InfostateTree> MakeEfficientInfostateTree(
    const std::unique_ptr<State> &start_state, double chance_reach_prob,
    const std::shared_ptr<Observer> &infostate_observer, Player acting_player,
    int max_move_limit, int storage_policy) {
  return std::shared_ptr<InfostateTree>(
      new InfostateTree(start_state, chance_reach_prob, infostate_observer,
                        acting_player, max_move_limit, storage_policy));
}

std::shared_ptr<InfostateTree> MakeLiarsDiceInfostateTree(
    const std::unique_ptr<State> &start_state, double chance_reach_prob,
    const std::shared_ptr<Observer> &infostate_observer, Player acting_player,
    int max_move_limit, int storage_policy) {
  return std::shared_ptr<InfostateTree>(
      new InfostateTree(start_state, chance_reach_prob, infostate_observer,
                        acting_player, max_move_limit, storage_policy, true));
}

std::shared_ptr<InfostateTree>
MakeInfostateTree(const std::vector<const State *> &start_states,
                  const std::vector<double> &chance_reach_probs,
                  std::shared_ptr<Observer> infostate_observer,
                  Player acting_player, int max_move_ahead_limit,
                  int storage_policy) {
  return std::shared_ptr<InfostateTree>(
      new InfostateTree(start_states, chance_reach_probs, infostate_observer,
                        acting_player, max_move_ahead_limit, storage_policy));
}

std::shared_ptr<InfostateTree>
MakeInfostateTree(const std::vector<std::unique_ptr<State>> &start_states,
                  const std::vector<double> &chance_reach_probs,
                  std::shared_ptr<Observer> infostate_observer,
                  Player acting_player, int max_move_ahead_limit,
                  int storage_policy) {
  return std::shared_ptr<InfostateTree>(
      new InfostateTree(start_states, chance_reach_probs, infostate_observer,
                        acting_player, max_move_ahead_limit, storage_policy));
}

std::shared_ptr<InfostateTree> MakeResolvingInfostateTree(
    const std::vector<std::unique_ptr<State>> &start_states,
    const std::vector<double> &chance_reach_probs,
    std::shared_ptr<Observer> infostate_observer, Player acting_player,
    Player ft_player,
    const std::unordered_map<std::string, double> &cf_value_constraints,
    int max_move_ahead_limit, int storage_policy) {
  SPIEL_CHECK_FALSE(cf_value_constraints.empty());
  return std::shared_ptr<InfostateTree>(new InfostateTree(
      start_states, chance_reach_probs, infostate_observer, acting_player,
      max_move_ahead_limit, storage_policy, cf_value_constraints, ft_player));
}

std::vector<std::shared_ptr<InfostateTree>> MakeResolvingInfostateTrees(
    const std::vector<std::unique_ptr<State>> &start_states,
    const std::vector<double> &chance_reach_probs,
    std::shared_ptr<Observer> infostate_observer, Player ft_player,
    const std::unordered_map<std::string, double> &cf_value_constraints,
    int max_move_ahead_limit, int storage_policy) {
  SPIEL_CHECK_FALSE(start_states.empty());
  const Game &game = *start_states[0]->GetGame();
  std::vector<std::shared_ptr<InfostateTree>> trees;
  trees.reserve(game.NumPlayers());
  for (int pl = 0; pl < game.NumPlayers(); ++pl) {
    trees.push_back(MakeResolvingInfostateTree(
        start_states, chance_reach_probs, infostate_observer, pl, ft_player,
        cf_value_constraints, max_move_ahead_limit, storage_policy));
  }
  return trees;
}

std::vector<std::shared_ptr<InfostateTree>>
MakeInfostateTrees(const Game &game, int max_move_ahead_limit,
                   int storage_policy) {
  std::vector<std::shared_ptr<InfostateTree>> trees;
  trees.reserve(game.NumPlayers());
  for (int pl = 0; pl < game.NumPlayers(); ++pl) {
    trees.push_back(
        MakeInfostateTree(game, pl, max_move_ahead_limit, storage_policy));
  }
  return trees;
}

std::vector<std::shared_ptr<InfostateTree>>
MakeInfostateTrees(const std::vector<const State *> &start_states,
                   const std::vector<double> &chance_reach_probs,
                   std::shared_ptr<Observer> infostate_observer,
                   int max_move_ahead_limit, int storage_policy) {
  std::vector<std::shared_ptr<InfostateTree>> trees;
  const Game &game = *start_states[0]->GetGame();
  trees.reserve(game.NumPlayers());
  for (int pl = 0; pl < game.NumPlayers(); ++pl) {
    trees.push_back(MakeInfostateTree(start_states, chance_reach_probs,
                                      infostate_observer, pl,
                                      max_move_ahead_limit, storage_policy));
  }
  return trees;
}

std::vector<std::shared_ptr<InfostateTree>>
MakeEfficientInfostateTrees(const std::unique_ptr<State> &start_state,
                            double chance_reach_prob,
                            std::shared_ptr<Observer> infostate_observer,
                            int max_move_ahead_limit, int storage_policy) {
  std::vector<std::shared_ptr<InfostateTree>> trees;
  const Game &game = *start_state->GetGame();
  trees.reserve(game.NumPlayers());
  for (int pl = 0; pl < game.NumPlayers(); ++pl) {
    trees.push_back(MakeEfficientInfostateTree(
        start_state, chance_reach_prob, infostate_observer, pl,
        max_move_ahead_limit, storage_policy));
  }
  return trees;
}

std::vector<std::shared_ptr<InfostateTree>>
MakeLiarsDiceInfostateTrees(const std::unique_ptr<State> &start_state,
                            double chance_reach_prob,
                            std::shared_ptr<Observer> infostate_observer,
                            int max_move_ahead_limit, int storage_policy) {
  std::vector<std::shared_ptr<InfostateTree>> trees;
  const Game &game = *start_state->GetGame();
  const liars_dice::LiarsDiceGame &liars_dice_game =
          down_cast<const liars_dice::LiarsDiceGame &>(*start_state->GetGame());
  for(int i = 0; i < liars_dice_game.total_num_dice(); i++) {
    start_state->ApplyAction(0);
  }
  trees.reserve(game.NumPlayers());
  for (int pl = 0; pl < game.NumPlayers(); ++pl) {
    trees.push_back(MakeLiarsDiceInfostateTree(
        start_state, chance_reach_prob, infostate_observer, pl,
        max_move_ahead_limit, storage_policy));
  }
  return trees;
}

std::vector<std::shared_ptr<InfostateTree>>
MakeInfostateTrees(const std::vector<std::unique_ptr<State>> &start_states,
                   const std::vector<double> &chance_reach_probs,
                   std::shared_ptr<Observer> infostate_observer,
                   int max_move_ahead_limit, int storage_policy) {
  std::vector<std::shared_ptr<InfostateTree>> trees;
  const Game &game = *start_states[0]->GetGame();
  trees.reserve(game.NumPlayers());
  for (int pl = 0; pl < game.NumPlayers(); ++pl) {
    trees.push_back(MakeInfostateTree(start_states, chance_reach_probs,
                                      infostate_observer, pl,
                                      max_move_ahead_limit, storage_policy));
  }
  return trees;
}

std::vector<std::shared_ptr<InfostateTree>>
MakePokerInfostateTrees(const std::unique_ptr<State> &start_state,
                        const std::vector<double> &chance_reach_probs,
                        const std::shared_ptr<Observer> &infostate_observer,
                        int round_limit, int storage_policy,
                        const std::vector<int> &board_cards) {
  std::vector<std::shared_ptr<InfostateTree>> trees;
  const Game &game = *start_state->GetGame();
  trees.reserve(game.NumPlayers());
  for (int pl = 0; pl < game.NumPlayers(); ++pl) {
    trees.push_back(MakePokerInfostateTree(start_state, chance_reach_probs,
                                           infostate_observer, pl, board_cards,
                                           round_limit, storage_policy));
  }
  return trees;
}

SequenceId InfostateTree::empty_sequence() const {
  return root().sequence_id();
}
absl::optional<DecisionId>
InfostateTree::DecisionIdForSequence(const SequenceId &sequence_id) const {
  SPIEL_DCHECK_TRUE(sequence_id.BelongsToTree(this));
  InfostateNode *node = sequences_.at(sequence_id.id());
  SPIEL_DCHECK_TRUE(node);
  if (node->is_root_node()) {
    return {};
  } else {
    return node->parent_->decision_id();
  }
}
absl::optional<InfostateNode *>
InfostateTree::DecisionForSequence(const SequenceId &sequence_id) {
  SPIEL_DCHECK_TRUE(sequence_id.BelongsToTree(this));
  InfostateNode *node = sequences_.at(sequence_id.id());
  SPIEL_DCHECK_TRUE(node);
  if (node->is_root_node()) {
    return {};
  } else {
    return node->parent_;
  }
}
bool InfostateTree::IsLeafSequence(const SequenceId &sequence_id) const {
  SPIEL_DCHECK_TRUE(sequence_id.BelongsToTree(this));
  InfostateNode *node = sequences_.at(sequence_id.id());
  SPIEL_DCHECK_TRUE(node);
  return node->start_sequence_id() == node->end_sequence_id();
}
std::vector<DecisionId>
InfostateTree::DecisionIdsWithParentSeq(const SequenceId &sequence_id) const {
  std::vector<DecisionId> out;
  const InfostateNode *observation_node = sequences_.at(sequence_id.id());
  std::stack<const InfostateNode *> open_set;
  for (const InfostateNode *child : observation_node->child_iterator()) {
    open_set.push(child);
  }
  while (!open_set.empty()) {
    const InfostateNode *node = open_set.top();
    open_set.pop();
    if (node->type() == kDecisionInfostateNode &&
        node->sequence_id() == sequence_id) {
      out.push_back(node->decision_id());
    } else {
      for (const InfostateNode *child : node->child_iterator()) {
        open_set.push(child);
      }
    }
  }
  return out;
}

void InfostateTree::LabelNodesWithIds() {
  // Idea of labeling: label the leaf sequences first, and continue up the tree.
  size_t sequence_index = 0;
  size_t decision_index = 0;

  // Do not label leaf nodes with sequences.
  const int start_depth = nodes_at_depths_.size() - 2;

  for (int depth = start_depth; depth >= 0; --depth) {
    for (InfostateNode *node : nodes_at_depths_[depth]) {
      if (node->type() != kDecisionInfostateNode)
        continue;
      decision_infostates_.push_back(node);
      node->decision_id_ = DecisionId(decision_index++, this);

      for (InfostateNode *child : node->child_iterator()) {
        sequences_.push_back(child);
        child->sequence_id_ = SequenceId(sequence_index++, this);
      }
      // We could use sequence_index to set start and end sequences for
      // the decision infostate right away here, however we'd like to make
      // sure to label correctly all nodes in the tree.
    }
  }
  // Finally label the last sequence (an empty sequence) in the root node.
  sequences_.push_back(mutable_root());
  mutable_root()->sequence_id_ = SequenceId(sequence_index, this);

  CollectStartEndSequenceIds(mutable_root(), mutable_root()->sequence_id());
}

// Make a recursive call to assign the parent's sequences appropriately.
// Collect pairs of (start, end) sequence ids from children and propagate
// them up the tree. In case that deep nodes (close to the leaves) do not
// have any child decision nodes, set the (start, end) to the parent sequence.
// In this way the range iterator will be empty (start==end) and well defined.
std::pair<size_t, size_t>
InfostateTree::CollectStartEndSequenceIds(InfostateNode *node,
                                          const SequenceId parent_sequence) {
  size_t min_index = kUndefinedNodeId; // This is a large number.
  size_t max_index = 0;
  const SequenceId propagate_sequence_id =
      node->sequence_id_.is_undefined()
          ? parent_sequence
          : node->sequence_id(); // This becomes the parent for next nodes.

  for (InfostateNode *child : node->child_iterator()) {
    auto [min_child, max_child] =
        CollectStartEndSequenceIds(child, propagate_sequence_id);
    min_index = std::min(min_child, min_index);
    max_index = std::max(max_child, max_index);
  }

  if (min_index != kUndefinedNodeId) {
    SPIEL_CHECK_LE(min_index, max_index);
    node->start_sequence_id_ = SequenceId(min_index, this);
    node->end_sequence_id_ = SequenceId(max_index + 1, this);
  } else {
    node->start_sequence_id_ = propagate_sequence_id;
    node->end_sequence_id_ = propagate_sequence_id;
  }

  if (node->sequence_id_.is_undefined()) {
    // Propagate children limits.
    node->sequence_id_ = parent_sequence;
    return {min_index, max_index};
  } else {
    // We have hit a defined sequence id, propagate it up.
    return {node->sequence_id_.id(), node->sequence_id_.id()};
  }
}

std::pair<double, SfStrategy>
InfostateTree::BestResponse(TreeplexVector<double> &&gradient) const {
  SPIEL_CHECK_EQ(this, gradient.tree());
  SPIEL_CHECK_EQ(num_sequences(), gradient.size());
  SfStrategy response(this);

  // 1. Compute counterfactual best response
  // (i.e. in all infostates, even unreachable ones)
  SequenceId current(0, this);
  const double init_value = -std::numeric_limits<double>::infinity();
  while (current.id() <= empty_sequence().id()) {
    double max_value = init_value;
    SequenceId max_id = current;
    const InfostateNode *node = observation_infostate(current);
    for (current = node->start_sequence_id();
         current != node->end_sequence_id(); current.next()) {
      if (gradient[current] > max_value) {
        max_value = gradient[current];
        max_id = current;
      }
    }
    if (init_value != max_value) {
      gradient[node->sequence_id()] += max_value;
      response[max_id] = 1.;
    }
    current.next();
  }
  SPIEL_CHECK_EQ(current.id(), empty_sequence().id() + 1);

  // 2. Prune away unreachable subtrees.
  //
  // This can be done with a more costly recursion.
  // Instead we make a more cache-friendly double pass through the response
  // vector: we increment the visited path by 1, resulting in a value of 2.
  // Then we zero-out all values but 2.
  current = empty_sequence();
  response[current] = 2.;
  while (!IsLeafSequence(current)) {
    for (SequenceId seq : observation_infostate(current)->AllSequenceIds()) {
      if (response[seq] == 1.) {
        current = seq;
        response[seq] += 1.;
        break;
      }
    }
  }
  for (SequenceId seq : response.range()) {
    response[seq] = response[seq] == 2. ? 1. : 0.;
  }
  SPIEL_DCHECK_TRUE(IsValidSfStrategy(response));
  return {gradient[empty_sequence()], response};
}

double InfostateTree::BestResponseValue(LeafVector<double> &&gradient) const {
  // Loop over all heights.
  for (int d = tree_height_ - 1; d >= 0; d--) {
    int left_offset = 0;
    // Loop over all parents of current nodes.
    for (int parent_idx = 0; parent_idx < nodes_at_depths_[d].size();
         parent_idx++) {
      const InfostateNode *node = nodes_at_depths_[d][parent_idx];
      const int num_children = node->num_children();
      const Range<LeafId> children_range =
          gradient.range(left_offset, left_offset + num_children);
      const LeafId parent_id(parent_idx, this);

      if (node->type() == kDecisionInfostateNode) {
        double max_value = std::numeric_limits<double>::min();
        for (LeafId id : children_range) {
          max_value = std::fmax(max_value, gradient[id]);
        }
        gradient[parent_id] = max_value;
      } else {
        SPIEL_DCHECK_EQ(node->type(), kObservationInfostateNode);
        double sum_value = 0.;
        for (LeafId id : children_range) {
          sum_value += gradient[id];
        }
        gradient[parent_id] = sum_value;
      }
      left_offset += num_children;
    }
    // Check that we passed over all of the children.
    SPIEL_DCHECK_EQ(left_offset, nodes_at_depths_[d + 1].size());
  }
  const LeafId root_id(0, this);
  return gradient[root_id];
}

DecisionId InfostateTree::DecisionIdFromInfostateString(
    const std::string &infostate_string) const {
  for (InfostateNode *node : decision_infostates_) {
    if (node->infostate_string() == infostate_string)
      return node->decision_id();
  }
  return kUndefinedDecisionId;
}

const InfostateNode *InfostateTree::DecisionNodeFromInfostateString(
    const std::string &infostate_string) const {
  for (InfostateNode *node : decision_infostates_) {
    if (node->infostate_string() == infostate_string)
      return node;
  }
  return nullptr;
}

bool CheckSum(const SfStrategy &strategy, SequenceId id, double expected_sum) {
  if (fabs(strategy[id] - expected_sum) > 1e-13) {
    return false;
  }

  const InfostateTree *tree = strategy.tree();
  if (tree->IsLeafSequence(id)) {
    return true;
  }

  double actual_sum = 0.;
  const InfostateNode *node = tree->observation_infostate(id);
  for (SequenceId sub_seq : node->AllSequenceIds()) {
    actual_sum += strategy[sub_seq];
  }
  if (fabs(actual_sum - expected_sum) > 1e-13) {
    return false;
  }

  for (SequenceId sub_seq : node->AllSequenceIds()) {
    if (!CheckSum(strategy, sub_seq, strategy[sub_seq])) {
      return false;
    }
  }
  return true;
}

bool IsValidSfStrategy(const SfStrategy &strategy) {
  return CheckSum(strategy, strategy.tree()->empty_sequence(), 1.);
}

} // namespace algorithms
} // namespace open_spiel
