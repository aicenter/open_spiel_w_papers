#include "algorithms/tabular_exploitability.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "subgame.h"
#include "algorithms/cfr.h"
#include "algorithms/best_response.h"
#include "infostate_tree_br.h"
#include "open_spiel/algorithms/poker_data.h"
#include "algorithms/expected_returns.h"
#include "algorithms/get_all_states.h"

#include <iostream>

namespace open_spiel{
    void CheckObserverAgainstDefaultObservations() {
        std::string game_name = "liars_dice(dice_sides=3,numdice=2)";
        std::shared_ptr<const Game> game = LoadGame(game_name);

        auto start = std::chrono::high_resolution_clock::now();
        std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
        std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

        auto states = algorithms::GetAllStates(*game, -1, /*include_terminals=*/false, /*include_chance_states=*/false);

        Observation public_observation(*game, public_observer);
        Observation infostate_observation(*game, infostate_observer);

        for (auto& [key, state] : states) {
            public_observation.SetFrom(*state, state->CurrentPlayer());
            infostate_observation.SetFrom(*state, state->CurrentPlayer());
            absl::Span<float> tensor = infostate_observation.Tensor();
            std::vector<float> infostate_observer_tensor(tensor.begin(), tensor.end());

            std::vector<float> old_infostate_tensor = state->InformationStateTensor();

            if (infostate_observer_tensor.size() != old_infostate_tensor.size()) {
                std::cout << "Vectors have different sizes!\n";
            } else {
                for (size_t i = 0; i < infostate_observer_tensor.size(); ++i) {
                    if (infostate_observer_tensor[i] != old_infostate_tensor[i]) {
                        std::cout << "Difference at index " << i
                                << ": " << infostate_observer_tensor[i]
                                << " vs " << old_infostate_tensor[i] << "\n";
                    }
                }
            }
        }

        for (auto& [key, state] : states) {
            std::string observer_string = infostate_observation.StringFrom(*state, state->CurrentPlayer());

            std::string old_string = state->InformationStateString();

            if (observer_string.size() != old_string.size()) {
                std::cout << "Strings have different sizes!\n";
                std::cout << "Observer string: " << observer_string << std::endl;
                std::cout << "Old infostate string: " << old_string << std::endl;
            } else {
                for (size_t i = 0; i < observer_string.size(); ++i) {
                    if (observer_string[i] != old_string[i]) {
                        std::cout << "Difference at index " << i
                                << ": " << observer_string[i]
                                << " vs " << old_string[i] << "\n";
                    }
                }
            }
        }
    }
}
int main(int argc, char **argv) {   
    open_spiel::CheckObserverAgainstDefaultObservations();
    
}
