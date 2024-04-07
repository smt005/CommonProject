// ◦ Xyz ◦
#include "Actions.h"
#include "../Functions.h"

#include <Object/Model.h>
#include <Object/Color.h>
#include "../Events.h"

namespace commands{
	void FadeModel(const std::string& modelName, float fadeTime)
	{
		if (Model::Ptr& model = Model::getByName(modelName)) {
			std::string idObserver = std::string("FadeModel:" + modelName);
			float dAlpha = 1.f / fadeTime;

			EventOnUpdate::Instance().Add(idObserver, [idObserver, model, dAlpha]() {
				float alpha = model->getAlpha();
				alpha -= dAlpha;
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
}
