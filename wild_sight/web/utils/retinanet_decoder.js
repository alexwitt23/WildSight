import * as tf from '@tensorflow/tfjs'
import json from '../public/anchors.json'


class RetinaNetDecoder {
    constructor(
        num_classes = 1,
        score_threshold = 0.4,
        topk_candidates = 100,
        nms_threshold = 0.3,
        max_detections_per_image = 10,
    ) {
        this.num_classes = num_classes
        this.score_threshold = score_threshold
        this.topk_candidates = topk_candidates
        this.nms_threshold = nms_threshold
        this.max_detections_per_image = max_detections_per_image
        
        // Create the anchors tensor.
        let anchors_arr = []
        for (var key of Object.keys(json)) {
            anchors_arr.push(json[key])
        }
        this.anchors = tf.tensor(anchors_arr, [anchors_arr.length, 4], "float32")
    }

    async get_boxes(classifications, regressions){

        // Send classification logits to confidences
        classifications = classifications.reshape([-1]).sigmoid()

        // Get the topk scoring boxes
        var {values, indices} = tf.topk(classifications, this.topk_candidates, true)

        // Gather the matched anchors. 
        const anchor_idxs = indices.floorDiv(this.num_classes)
        let classes_idxs = indices.mod(this.num_classes)
        const anchors_topk = this.anchors.gather(anchor_idxs)

        // Get widths, heights, and centers of anchor boxes
        const widths = anchors_topk.slice([0, 2], [-1, 1]).sub(anchors_topk.slice([0, 0], [-1, 1]))
        const heights = anchors_topk.slice([0, 3], [-1, 1]).sub(anchors_topk.slice([0, 1], [-1, 1]))
        const ctr_x = anchors_topk.slice([0, 0] ,[-1, 1]).add(tf.tensor([0.5]).mul(widths))
        const ctr_y = anchors_topk.slice([0, 1] ,[-1, 1]).add(tf.tensor([0.5]).mul(heights))
        
        const regressions_topk = regressions.gather(anchor_idxs)
        const dx = regressions_topk.slice([0, 0], [-1, 1])
        const dy = regressions_topk.slice([0, 1], [-1, 1])
        const dw = regressions_topk.slice([0, 2] ,[-1, 1])
        const dh = regressions_topk.slice([0, 3] ,[-1, 1])
        
        const pred_ctr_x = dx.mul(widths).add(ctr_x)
        const pred_ctr_y = dy.mul(heights).add(ctr_y)
        const pred_w = tf.exp(dw).mul(widths)
        const pred_h = tf.exp(dh).mul(heights)
        // TODO(alex): this is flipped from xyxy to yxyx and seems to work. Why?
        var predictions = tf.concat([
          pred_ctr_x.sub(tf.tensor([0.5]).mul(pred_w)),
          pred_ctr_y.sub(tf.tensor([0.5]).mul(pred_h)),
          pred_ctr_x.add(tf.tensor([0.5]).mul(pred_w)),
          pred_ctr_y.add(tf.tensor([0.5]).mul(pred_h)),
        ], -1)
        var nms_keep = await tf.image.nonMaxSuppressionAsync(
            predictions, values, this.max_detections_per_image, this.nms_threshold, this.score_threshold
        )
        const bboxes = predictions.gather(nms_keep)
        const confidences = values.gather(nms_keep)
        classes_idxs = classes_idxs.gather(nms_keep)
        
        return [classes_idxs, bboxes, confidences]
    }

}


export {RetinaNetDecoder}