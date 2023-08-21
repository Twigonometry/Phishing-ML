const mongoose = require("mongoose");

const Schema = mongoose.Schema;

const EmailSchema = new Schema({
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

const EmailModel = mongoose.model("EmailModel", EmailSchema);