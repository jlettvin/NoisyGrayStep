# NoisyGrayStep## gray.pyThis creates static images illustrating how perceptionof a sharp boundary is hardly affected by image noise.<table><tr><td valign="top"><img src="https://github.com/jlettvin/NoisyGrayStep/blob/master/CleanGrayStep.png"><br />Two slightly different gray fields</img><br /><ul><lh>Observations:</lh><li>The average brightness is a neutral gray</li><li>The step in brightness is small</li><li>The boundary is clear and sharp</li></ul></td><td valign="top"><img src="https://github.com/jlettvin/NoisyGrayStep/blob/master/NoisyGrayStep.png"><br />Same fields with dominant noise:</img><br /><ul><lh>Observations:</lh><li>The average brightness is the same neutral gray</li><li>The step in brightness is the same small</li><li>The introduced noise has an RMS greater than the boundary step</li><li>The boundary is slightly less clear but still sharp</li></ul></td></tr></table>## egray.pyThis creates similar static images illustrating howintroduction of a gradient sharpens perception.## ungray.pyThis uses your first camera to acquire images which are thenprocessed to gray-level boundaries in all three color planes.