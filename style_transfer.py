from image_loader import *

from metrics_tracker import * 

from model_data import *

import tensorflow as tf

class StyleTransferModel:

    def __init__(self, model, iterations=1000, content_weight=1e3, style_weight=1e-2, learning_rate=5.0, epsilon=1e-1):
        self.iterations = iterations
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)

        self.model = self._initialize_model(model)

        self.content_layers = model.content_layers
        self.style_layers = model.style_layers

        self.model_module = model.model_module
        
        self.metrics = MetricsTracker()
    
    def _get_model_layers(self, model, layers: list[str]):
        return [ model.get_layer(name).output for name in layers ]

    def _initialize_model(self, model: ModelData):
        base_model = model.base_model
        base_model.trainable = False

        content_outputs = self._get_model_layers(base_model, model.content_layers)
        style_outputs = self._get_model_layers(base_model, model.style_layers)

        model = tf.keras.Model(base_model.input, style_outputs + content_outputs)
        
        for layer in model.layers:
            layer.trainable = False
        
        return model
    
    def _extract_feature_representations(self, style_image: ImageLoader, content_image: ImageLoader, num_style_layers: int):
        style_outputs = self.model(style_image.process_image_for_model(self.model_module))
        content_outputs = self.model(content_image.process_image_for_model(self.model_module))

        style_features = [ layer[0] for layer in style_outputs[:num_style_layers] ]
        content_features = [ layer[0] for layer in content_outputs[num_style_layers:] ]
        
        return style_features, content_features
    
    def _apply_gram_matrix(self, matrix_input):
        channels = int(matrix_input.shape[-1])

        matrix = tf.reshape(matrix_input, [-1, channels])
        gram = tf.matmul(matrix, matrix, transpose_a=True)

        return gram / tf.cast(tf.shape(matrix)[0], tf.float32)
    
    def _compute_loss_from_layers(self, features, output_features, loss_calculator, num_layers):
        weight_per_layer = 1.0 / num_layers
        total_loss = 0
        
        for target, comb in zip(features, output_features):
            total_loss += weight_per_layer * loss_calculator(comb[0], target)
        
        return total_loss 
    
    def _compute_style_loss(self, base_style, gram_target):
        gram_style = self._apply_gram_matrix(base_style)
        return self._compute_content_loss(gram_style, gram_target)
    
    def _compute_content_loss(self, base_content, target):
        square_result = tf.square(base_content - target)
        return tf.reduce_mean(square_result)

    def _compute_loss(self, initial_image, gram_style_features, content_features):
        outputs = self.model(initial_image)
        
        style_output_features = outputs[:len(self.style_layers)]
        content_output_features = outputs[len(self.style_layers):]

        style_loss = self.style_weight * self._compute_loss_from_layers(gram_style_features, style_output_features, self._compute_style_loss, len(self.style_layers))
        content_loss = self.content_weight * self._compute_loss_from_layers(content_features, content_output_features, self._compute_content_loss, len(self.content_layers))

        return style_loss + content_loss

    def _compute_gradient(self, initial_image, gram_style_features, content_features):
        with tf.GradientTape() as tape:
            total_loss = self._compute_loss(initial_image, gram_style_features, content_features)
        
        return tape.gradient(total_loss, initial_image), total_loss

    def apply_style_transfer(self, content_image: ImageLoader, style_image: ImageLoader):
        style_features, content_features = self._extract_feature_representations(style_image, content_image, len(self.style_layers))

        gram_style_features = [ self._apply_gram_matrix(feature) for feature in style_features ]

        init_content_image = tf.Variable(content_image.process_image_for_model(self.model_module), dtype=tf.float32)

        image_progress = []

        # TODO: remove hard coding the rows and cols of the progression pictures
        display_interval = self.iterations / (2 * 5)

        for i in range(self.iterations):
            print(i)
            # compute grads
            grads, total_loss = self._compute_gradient(init_content_image, gram_style_features, content_features)

            # apply gradients 
            self.optimizer.apply_gradients([(grads, init_content_image)])

            # clip images
            clipped_image = tf.clip_by_value(init_content_image, -content_image.normalized_means, 255 - content_image.normalized_means)
            
            # assign image 
            init_content_image.assign(clipped_image)
            
            self.metrics.track_best_loss_and_image(total_loss, content_image.clip_image_from_model(init_content_image.numpy()))
            
            self.metrics.track_metrics(content_image.get_image_array(), init_content_image.numpy())

            # display image
            if i % display_interval == 0:
                current_image = content_image.clip_image_from_model(init_content_image.numpy())
                image_progress.append(current_image)
        
        return self.metrics.best_image, self.metrics.best_loss, image_progress, self.metrics.get_metrics()