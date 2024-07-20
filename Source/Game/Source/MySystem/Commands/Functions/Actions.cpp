// ◦ Xyz ◦

#include "Actions.h"
#include <Core.h>
#include "../Functions.h"
#include "../../Quests/Quest.h"
#include "../../Quests/QuestManager.h"
#include <Object/Model.h>
#include <Object/Color.h>
#include "../Events.h"

namespace commands
{
	void FadeModel(const std::string& modelName, float fadeTime)
	{
		if (Model::Ptr& model = Model::getByName(modelName)) {
			std::string idObserver = std::string("FadeModel:" + modelName);

			EventOnUpdate::Instance().Add(idObserver, [idObserver, model, kAlpha = float(1000.f / fadeTime)]() {
				float alpha = model->getAlpha();
				alpha -= kAlpha * (float)Engine::Core::deltaTime();
				alpha = alpha < 0.f ? 0.f : alpha;
				model->setAlpha(alpha);

				if (alpha <= 0) {
					model->setAlpha(1.f);
					HideImage(model->getName());
					EventOnUpdate::Instance().Remove(idObserver);
				}
			});
		}
	}

	void DelayActionCommand(const std::string& questName, const std::string& commandsName, float delay)
	{
		if (Quest::Ptr questPtr = QuestManager::Instance().GetQuest(questName)) {
			Commands& commandMap = questPtr->_commandMap[commandsName];

			if (!commandMap.empty()) {
				std::string idObserver = std::string("DelayActionCommand:" + questPtr->Name() + commandsName);
				EventOnUpdate::Instance().Add(idObserver, [idObserver, commandMap, delayConst = (delay / 1000.f)]() {
					(*const_cast<float*>(&delayConst)) -= (float)Engine::Core::deltaTime();

					if (delayConst <= 0.f) {
						CommandManager::Run(commandMap);
						EventOnUpdate::Instance().Remove(idObserver);
					}
				});
			}
		}
	}
}
