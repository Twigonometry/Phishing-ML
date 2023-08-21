var express = require('express');
var router = express.Router();
const EmailsModel = require("../models/email");
const mongoose = require("mongoose");

/* GET home page. */
router.get('/', async function(req, res, next) {
  try {
    var email = await(EmailsModel.findOne({rating: null}).exec());
    console.log(email);
    res.render('index', { title: 'Data Labelling Portal', email: email });
  } catch (error) {
    return next(error);
  }
});

router.post('/rate', async function(req, res, next) {
  try {
    const params = req.body;
    const id = new mongoose.Types.ObjectId(params.id);

    const rating = {
      authoritative: params.authoritative,
      threatening: params.threatening,
      rewarding: params.rewarding,
      unnatural: params.unnatural,
      emotional: params.emotional,
      provoking: params.provoking,
      timesensitive: params.timesensitive,
      imperative: params.imperative
    }

    console.log(rating);

    await EmailsModel.findByIdAndUpdate(id, {rating: rating});

    res.redirect('/')
  } catch (error) {
    return next(error);
  }
})

module.exports = router;
