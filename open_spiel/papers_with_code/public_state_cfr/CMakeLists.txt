add_library(public_state_cfr OBJECT
        infostate_tree_br.cc
        infostate_tree_br.h
        subgame.cc
        subgame.h
        ../../algorithms/poker_data.h)

macro(add_paper_executable NAME SOURCES)
    add_executable(${NAME} ${SOURCES} $<TARGET_OBJECTS:public_state_cfr>
            ${OPEN_SPIEL_OBJECTS})
    target_link_libraries(${NAME})
endmacro()

add_paper_executable(my_main my_main.cc)
add_paper_executable(test_observer_liars_dice test_observer_liars_dice.cc)
