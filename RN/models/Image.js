// models/Image.js
import mongoose from 'mongoose';

const ImageSchema = new mongoose.Schema({
  name: String,
  data: String,  // Changed from Buffer to String to store base64
  contentType: String,
  uploadDate: {
    type: Date,
    default: Date.now
  },
  metadata: {
    originalName: String,
    size: Number,
    analysis: {
      status: {
        type: String,
        enum: ['pending', 'completed', 'failed'],
        default: 'pending'
      },
      results: mongoose.Schema.Types.Mixed
    }
  }
});

export const Image = mongoose.model('Image', ImageSchema);