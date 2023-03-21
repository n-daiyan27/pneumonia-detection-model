from keras.models import Model
from keras.layers import Flatten,Dense
from keras.applications.vgg16 import VGG16 
import matplotlib.pyplot as plot
from glob import glob

IMAGESHAPE = [224, 224, 3] 
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)
training_data = 'chest_xray/train_contour'
testing_data = 'chest_xray/test' 

for each_layer in vgg_model.layers:
	each_layer.trainable = False 
classes = glob('chest_xray/train_contour/*') 
flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)
final_model = Model(inputs=vgg_model.input, outputs=prediction)
final_model.summary() 
final_model.compile( 
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy']
)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
								shear_range = 0.2,
								zoom_range = 0.2,
								horizontal_flip = True,
								vertical_flip=True,
        						brightness_range=[0.5, 1.5] )
testing_datagen = ImageDataGenerator(rescale = 1./255,
								shear_range = 0.2,
								zoom_range = 0.2,
								horizontal_flip = True,
								vertical_flip=True,
        						brightness_range=[0.5, 1.5] )
training_set = train_datagen.flow_from_directory('chest_xray/train_contour',
												target_size = (224, 224),
												batch_size = 25,
												class_mode = 'categorical')
val_set = testing_datagen.flow_from_directory('chest_xray/test',
											target_size = (224, 224),
											batch_size = 25,
											class_mode = 'categorical')
fitted_model = final_model.fit(
training_set,
validation_data=val_set,
epochs=10,
steps_per_epoch=len(training_set),
validation_steps=len(val_set)
)
plot.plot(fitted_model.history['loss'], label='training loss') 
plot.plot(fitted_model.history['val_loss'], label='validation loss')
plot.legend()
plot.show()
plot.savefig('LossVal_loss')
plot.plot(fitted_model.history['accuracy'], label='training accuracy')
plot.plot(fitted_model.history['val_accuracy'], label='validation accuracy')
plot.legend()
plot.show()
plot.savefig('AccVal_acc')
final_model.save('our_model.h5') 
