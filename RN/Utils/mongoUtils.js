// utils/mongoUtils.js
import mongoose from 'mongoose';
import { Image } from '../models/Image';

export const connectDB = async (mongoUri) => {
  try {
    await mongoose.connect(mongoUri, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    console.log('MongoDB connected successfully');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    throw error;
  }
};

export const uriToBase64 = async (uri) => {
  try {
    if (uri.startsWith('file://')) {
      const response = await fetch(uri);
      const blob = await response.blob();
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }
    // If it's already a base64 string, return as is
    else if (uri.startsWith('data:')) {
      return uri;
    }
    throw new Error('Unsupported URI format');
  } catch (error) {
    console.error('Error converting URI to base64:', error);
    throw error;
  }
};

export const uploadImageToMongo = async (imageUri, mongoUri) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      await connectDB(mongoUri);
    }

    // Convert image to base64
    const base64Image = await uriToBase64(imageUri);

    const newImage = new Image({
      name: `image_${Date.now()}`,
      data: base64Image,
      contentType: 'image/jpeg',
      metadata: {
        originalName: `upload_${Date.now()}.jpg`,
        size: base64Image.length,
        analysis: {
          status: 'pending',
          results: null
        }
      }
    });

    const savedImage = await newImage.save();
    return savedImage;
  } catch (error) {
    console.error('Error uploading image:', error);
    throw error;
  }
};

export const getImagesFromMongo = async (mongoUri) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      await connectDB(mongoUri);
    }
    return await Image.find({}, { data: 0 });
  } catch (error) {
    console.error('Error retrieving images:', error);
    throw error;
  }
};