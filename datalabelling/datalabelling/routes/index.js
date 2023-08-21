var express = require('express');
var router = express.Router();
const EmailsModel = require("../models/email");
const mongoose = require("mongoose");

/* GET home page. */
router.get('/', async function(req, res, next) {
  try {
    var email = await(EmailsModel.findOne({rating: null}).exec());
    res.render('index', { title: 'Data Labelling Portal', email: email });
  } catch (error) {
    return next(error);
  }
});

router.post('/rate', async function(req, res, next) {
  try {
    const params = req.query;
    const id = new mongoose.Types.ObjectId(params.id);

    await EmailsModel.findByIdAndUpdate(id, {ratings: params.rating})

    res.redirect('/')
  } catch (error) {
    return next(error);
  }
})

module.exports = router;
