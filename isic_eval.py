#Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read RGB model checkpoints.""")
tf.app.flags.DEFINE_string('RGB_dir', '/home/charlie/cifar_isic_checkpoints/RGB_train',
                           """Directory where to read RGB model checkpoints.""")
tf.app.flags.DEFINE_string('FFT_dir', '/home/charlie/martin/cropped/FFT/cifar10_train',
                           """Directory where to read FFT model checkpoints.""")

tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 500,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op, tp,fp,tn,fn, guess, logits, model_ckpt):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_ckpt)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      print(ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      tp_count = 0
      fp_count = 0
      tn_count = 0
      fn_count = 0
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        guesses = sess.run([guess])
        logit_stream = sess.run([logits])
        predictions = sess.run([top_k_op])
        tp_pred = sess.run([tp])
        fp_pred = sess.run([fp])
        tn_pred = sess.run([tn])
        fn_pred = sess.run([fn])

        tp_count += np.sum(tp_pred)
        fp_count += np.sum(fp_pred)
        tn_count += np.sum(tn_pred)
        fn_count += np.sum(fn_pred)
        true_count += np.sum(predictions)

        step += 1

      precision = true_count / total_sample_count
      sensitivity = tp_count/(tp_count+fn_count)
      specificity = tn_count/(tn_count+fp_count)
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      print('%s: sensitivity @ 1 = %.3f' % (datetime.now(), sensitivity))
      print('%s: specificity @ 1 = %.3f' % (datetime.now(), specificity))
      print('%s: True Pos @ 1 = %i' % (datetime.now(), tp_count))
      print('%s: False Pos @ 1 = %i' % (datetime.now(), fp_count))
      print('%s: True Neg @ 1 = %i' % (datetime.now(), tn_count))
      print('%s: False Neg @ 1 = %i' % (datetime.now(), fn_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataflag):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = dataflag == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    guess = cifar10.inference(images)
    logits = tf.nn.softmax(guess)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    tp, fp, tn, fn = binary_score(logits, labels)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, tp,fp,tn,fn, guess, logits, FLAGS.FFT_dir)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def binary_score(logits,labels):
    is_label_one = tf.cast(labels, dtype=tf.bool)
    is_label_zero = tf.logical_not(is_label_one)
    correct_prediction = tf.nn.in_top_k(logits, labels, 1, name="correct_answers")
    false_prediction = tf.logical_not(correct_prediction)
    true_positives = tf.reduce_sum(tf.to_int32(tf.logical_and(correct_prediction,is_label_one)))
    false_positives = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_zero)))
    true_negatives = tf.reduce_sum(tf.to_int32(tf.logical_and(correct_prediction, is_label_zero)))
    false_negatives = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_one)))

    return true_positives, false_positives, true_negatives, false_negatives


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate(FLAGS.eval_data)


if __name__ == '__main__':
  tf.app.run()
