# NoisyGrayStep## gray.pyThis creates static images illustrating how perceptionof a sharp boundary is hardly affected by image noise.### Two slightly different gray fields:![alt Clean Gray Step image render failed](https://github.com/jlettvin/NoisyGrayStep/blob/master/CleanGrayStep.png)Observations:* The average brightness is a neutral gray* The step in brightness is small* The boundary is clear and sharp### Same fields with dominant noise:<b>HELLO</b>![alt Noisy Gray Step image render failed](https://github.com/jlettvin/NoisyGrayStep/blob/master/NoisyGrayStep.png)Observations:* The average brightness is the same neutral gray* The step in brightness is the same small* The introduced noise has an RMS greater than the boundary step* The boundary is slightly less clear but still sharp## egray.pyThis creates similar static images illustrating howintroduction of a gradient sharpens perception.## ungray.pyThis uses your first camera to acquire images which are thenprocessed to gray-level boundaries in all three color planes.