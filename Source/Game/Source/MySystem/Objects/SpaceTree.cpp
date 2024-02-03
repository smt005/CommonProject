#include "SpaceTree.h"
#include <algorithm>
#include <stdio.h>
#include <unordered_map>>
#include <set>
#include <Core.h>
#include "BodyData.h"

std::string SpaceTree::GetNameClass() {
	return Engine::GetClassName(this);
}

void SpaceTree::Update(double dt) {
	if (countOfIteration == 0 || _bodies.size() <= 1) {
		return;
	}

	for (size_t i = 0; i < countOfIteration; ++i) {
		Update();
	}
}

void SpaceTree::Update() {
}

void SpaceTree::Preparation() {
}
