import axios from "axios";
import React from 'react';

class QueryFormAndResponse extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            input_string: '',
            response_entities: [],
        };

        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleChange(event) {
        this.setState({input_string: event.target.value});
    }

    handleSubmit(event) {
        axios.post("/query", {
            "message": this.state.input_string,
            "b_optimized": true
        })
            .then(response => {
                this.setState({response_entities: response.data.response_entities});
            }).catch((error) => {
            if (error.response) {
                console.log(error.response)
                console.log(error.response.status)
                console.log(error.response.headers)
            }
        })

        event.preventDefault();
    }

    render() {
        return (
            <div className="bd-content ps-lg-2">
                <div className="d-grid gap-5">
                    <form onSubmit={this.handleSubmit}>
                        <div className="mb-3">
                            <label htmlFor="inputString" className="form-label"><b>Input String</b></label>
                            <input type="text" className="form-control" id="inputString" autoComplete="off"
                                   spellCheck="false" onChange={this.handleChange}/>
                        </div>
                        <button type="submit" className="btn btn-primary">Submit</button>
                    </form>
                    <table className="table table-hover">
                        <thead>
                        <tr>
                            <th scope="col">#</th>
                            <th scope="col">Segment</th>
                            <th scope="col">Score</th>
                        </tr>
                        </thead>
                        <tbody>
                        {
                            this.state.response_entities.map((response_entity, index) => (
                                <tr key={index}>
                                    <th scope="row">{index}</th>
                                    <td>{response_entity.segment}</td>
                                    <td>{response_entity.score}</td>
                                </tr>
                            ))
                        }
                        </tbody>
                    </table>
                </div>
            </div>
        );
    }
}

export default QueryFormAndResponse;