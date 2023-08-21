const mongoose = require("mongoose");

const Schema = mongoose.Schema;

const EmailSchema = new Schema({
  _id: mongoose.Schema.Types.ObjectId,
  body: String,
  ratings: {
    authoritative: Number,
    threatening: Number,
    rewarding: Number,
    unnatural: Number,
    emotional: Number,
    provoking: Number,
    timesensitive: Number,
    imperative: Number
  }
});

module.exports = mongoose.model("Email", EmailSchema)