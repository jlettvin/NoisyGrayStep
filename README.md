# NoisyGrayStep## gray.pyThis code creates static images illustrating howperception of boundary sharpness is affected little by noise.<table><tr><td valign="top"><img src="https://github.com/jlettvin/NoisyGrayStep/blob/master/CleanGrayStep.png"></img><br />Two slightly different gray fields<br /><ul><lh>Observations:</lh><li>average brightness is a neutral gray</li><li>step in brightness is small</li><li>boundary is clear and sharp</li></ul></td><td valign="top"><img src="https://github.com/jlettvin/NoisyGrayStep/blob/master/NoisyGrayStep.png"></img><br />Same fields with dominant noise:<br /><ul><lh>Observations:</lh><li>average brightness is the same</li><li>step in average brightness is the same</li><li>introduced noise greater than the step</li><li>boundary is less clear but still sharp</li></ul></td></tr></table>## egray.pyThis creates similar static images illustrating howintroduction of a gradient sharpens perception.## ungray.pyThis uses your first camera to acquire images which are thenprocessed to gray-level boundaries in all three color planes.